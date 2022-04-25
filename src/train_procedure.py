import torch
from torch.utils.data import (DataLoader, RandomSampler)
from torch.utils.data.distributed import DistributedSampler
from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule, WarmupConstantSchedule
import logging
import time
import copy

logger = logging.getLogger(__name__)


def train(args, train_dataset, eval_dataset, model, tokenizer):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, num_workers=args.workers, sampler=train_sampler,
                                  batch_size=args.train_batch_size) # , collate_fn=trim_batch)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    #train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    #set_seed(args.seed, args.n_gpu)  # Added here for reproductibility (even between python 2 and 3)

    best_score = 0
    best_model = {
        'epoch': 0,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict()
    }

    for epoch in range(int(args.num_train_epochs)):
    #for epoch in train_iterator:
        #epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        total_loss = 0
        total_norm = 0
        count_norm = 0

        t_start = time.time()
        for step, batch in enumerate(train_dataloader):
        #for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # XLM don't use segment_ids
                      'labels':         batch[3],
                      'img_feats':      None if args.img_feature_dim == -1 else batch[5]}
            outputs = model(**inputs)

            #loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            loss, logits = outputs[:2]

            if args.n_gpu > 1: loss = loss.mean() # mean() to average on multi-gpu parallel training

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                total_norm += torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                count_norm += 1

            #batch_score = compute_score_with_logits(logits, batch[4]).sum()
            #train_score += batch_score.item()

            tr_loss += loss.item()
            total_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:# Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        logger.info("Epoch: %d, global_step: %d" % (epoch, global_step))
                        eval_result, eval_score = evaluate(args, model, eval_dataset, prefix=global_step)
                        if eval_score > best_score:
                            best_score = eval_score
                            best_model['epoch'] = epoch
                            best_model['model'] = copy.deepcopy(model)

                        logger.info("EVALERR: {}%".format(100 * best_score))
                    logging_loss = tr_loss

            #if args.max_steps > 0 and global_step > args.max_steps:
            #    epoch_iterator.close()
            #    break

        t_end = time.time()
        logger.info('Train Time Cost: %.3f' % (t_end-t_start))

        # evaluation
        logger.info("Epoch: %d" % (epoch))
        eval_result, eval_score = evaluate(args, model, eval_dataset, prefix=global_step)
        if eval_score > best_score:
            best_score = eval_score
            best_model['epoch'] = epoch
            best_model['model'] = copy.deepcopy(model)
            #best_model['optimizer'] = copy.deepcopy(optimizer.state_dict())

        # save checkpoints
        if args.local_rank in [-1, 0] and args.save_epoch > 0 and epoch % args.save_epoch == 0: # Save model checkpoint
            output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(epoch))
            if not os.path.exists(output_dir): os.makedirs(output_dir)
            model_to_save = best_model['model'].module if hasattr(model, 'module') else best_model['model']  # Take care of distributed/parallel training

            save_num = 0
            while (save_num < 10):
                try:
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    tokenizer.save_pretrained(output_dir)
                    break
                except:
                    save_num += 1
            logger.info("Saving model checkpoint {0} to {1}".format(epoch, output_dir))

        epoch_log = {'epoch': epoch, 'eval_score': eval_score, 'best_score':best_score}
        log_json.append(epoch_log)

        with open(args.output_dir + '/eval_logs.json', 'w') as fp:
            json.dump(log_json, fp)

        logger.info("PROGRESS: {}%".format(round(100*(epoch + 1) / args.num_train_epochs, 4)))
        logger.info("EVALERR: {}%".format(100*best_score))
        logger.info("LOSS: {}%".format(total_loss / len(train_dataset)))

    with open(args.output_dir + '/eval_logs.json', 'w') as fp:
        json.dump(log_json, fp)

    if args.local_rank in [-1, 0]: # Save the final model checkpoint
        output_dir = os.path.join(args.output_dir, 'best-{}'.format(best_model['epoch']))
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        model_to_save = best_model['model'].module if hasattr(model, 'module') else best_model['model']  # Take care of distributed/parallel training

        save_num = 0
        while (save_num < 10):
            try:
                model_to_save.save_pretrained(output_dir)
                torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                tokenizer.save_pretrained(output_dir)
                break
            except:
                save_num += 1
        logger.info("Saving the best model checkpoint epoch {} to {}".format(best_model['epoch'], output_dir))

    return global_step, tr_loss / global_step
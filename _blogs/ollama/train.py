from dataclasses import asdict

import torch

from ollama.dataloader import get_dataset


def load_model(train_args):
    from liger_kernel.transformers import AutoLigerKernelForCausalLM

    model_kwargs = dict(
        use_cache=False,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=None,
    )

    model = AutoLigerKernelForCausalLM.from_pretrained(train_args.model, **model_kwargs)
    return model


def initialize_tokenizer(model_name):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.model_max_length = 2048
    tokenizer.pad_token = (
        tokenizer.unk_token
    )  # use unk rather than eos token to prevent endless generation
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    tokenizer.padding_side = "right"
    return tokenizer


def tokenize_data(tokenizer, examples):
    tokens = tokenizer(
        # add eos token to each example
        [f"{t}{tokenizer.eos_token}" for t in examples["text"]]
    )
    return tokens


def prepare_dataset(dataset_dir, train_args, tokenizer):
    from pathlib import Path

    dataset_dir.download()

    dataset = get_dataset(
        Path(dataset_dir.path.replace("file://", "")).expanduser(),
        num_proc=train_args.dataloader_num_proc,
        block_size=train_args.model_max_length,
        skip_by=train_args.model_max_length,
    ).map(
        lambda x: tokenize_data(tokenizer, x),
        batched=True,
        num_proc=train_args.dataloader_num_proc,
    )

    dataset_splits = dataset.train_test_split(
        test_size=train_args.test_size, seed=train_args.seed
    )
    return dataset_splits


def create_trainer(model, train_args, peft_args, dataset_splits, tokenizer):
    from peft import LoraConfig
    from transformers import DataCollatorForLanguageModeling, TrainingArguments
    from trl import SFTTrainer

    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = SFTTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=train_args.output_dir,
            overwrite_output_dir=train_args.overwrite_output_dir,
            do_eval=train_args.do_eval,
            evaluation_strategy="steps" if train_args.do_eval else "no",
            per_device_train_batch_size=train_args.per_device_train_batch_size,
            per_device_eval_batch_size=train_args.per_device_eval_batch_size,
            learning_rate=train_args.learning_rate,
            logging_steps=train_args.logging_steps,
            logging_strategy=train_args.logging_strategy,
            save_steps=train_args.save_steps,
            save_total_limit=train_args.save_total_limit,
            seed=train_args.seed,
            bf16=train_args.bf16,
            gradient_accumulation_steps=train_args.gradient_accumulation_steps,
            gradient_checkpointing=train_args.gradient_checkpointing,
            max_steps=train_args.max_steps,
            num_train_epochs=train_args.num_train_epochs,
            lr_scheduler_type=train_args.lr_scheduler_type,
            warmup_ratio=train_args.warmup_ratio,
            remove_unused_columns=train_args.remove_unused_columns,
            dataloader_num_workers=train_args.dataloader_num_proc,
        ),
        peft_config=LoraConfig(**asdict(peft_args)),
        data_collator=data_collator,
        train_dataset=dataset_splits["train"],
        eval_dataset=dataset_splits["test"],
        max_seq_length=2048,
        dataset_text_field="text",
        tokenizer=tokenizer,
        packing=True,
    )
    return trainer


def save_model(trainer, train_args):
    from peft import PeftModel

    trainer.train()
    trainer.save_model(train_args.adapter_dir)

    model = PeftModel.from_pretrained(trainer.model, train_args.adapter_dir)
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(train_args.output_dir)
    trainer.tokenizer.save_pretrained(train_args.output_dir)

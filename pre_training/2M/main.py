
import torch
from transformers import RobertaForMaskedLM, RobertaConfig
from DataCollatorForLanguageModeling import DataCollatorForLanguageModeling
from LineByLineTextDataset import LineByLineTextDataset
from transformers import Trainer, TrainingArguments
from ILtokenizer import SMILES_Atomwise_Tokenizer


tokenizer=SMILES_Atomwise_Tokenizer('updated_vocab.txt')



# 指定模型配置
config = RobertaConfig(
    vocab_size=1295,
    hidden_size=256,
    num_hidden_layers=3,
    num_attention_heads=4,
    intermediate_size=512,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1
)




model = RobertaForMaskedLM(config=config)

params = model.parameters()
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters in the model: {total_params}")




# block_size: max length of batch
traindataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="sampled_1.98M.txt",
    block_size=100,
    n_jobs=10,)

valdataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="sampled_0.02M.txt",
    block_size=100,
    n_jobs=10,)



data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)



# 更新训练参数
training_args = TrainingArguments(
    output_dir="./sigma",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    save_steps=2000,
    save_strategy= 'epoch',
    fp16=True,
    save_total_limit=2,
    learning_rate=1e-4,
    prediction_loss_only=False,
    report_to="tensorboard",
    logging_dir="./logs",

)

# 创建 Trainer 实例
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=traindataset,
    eval_dataset=valdataset,
)


trainer.train()
model = trainer.model
torch.save(model.state_dict(), "pretrained_model_2M.pth")





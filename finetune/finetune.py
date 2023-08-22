from torchkeras import KerasModel
from accelerate import Accelerator
from data_preprocess import *

class StepRunner:
    def __init__(self, net, loss_fn, accelerator=None, stage="train", metrics_dict=None,
                 optimizer=None, lr_scheduler=None
                 ):
        self.net, self.loss_fn, self.metrics_dict, self.stage = net, loss_fn, metrics_dict, stage
        self.optimizer, self.lr_scheduler = optimizer, lr_scheduler
        self.accelerator = accelerator if accelerator is not None else Accelerator()
        if self.stage == 'train':
            self.net.train()
        else:
            self.net.eval()

    def __call__(self, batch):

        # loss
        with self.accelerator.autocast():
            loss = self.net(input_ids=batch["input_ids"], labels=batch["labels"]).loss

        # backward()
        if self.optimizer is not None and self.stage == "train":
            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.net.parameters(), 1.0)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()

        all_loss = self.accelerator.gather(loss).sum()

        # losses (or plain metrics that can be averaged)
        step_losses = {self.stage + "_loss": all_loss.item()}

        # metrics (stateful metrics)
        step_metrics = {}

        if self.stage == "train":
            if self.optimizer is not None:
                step_metrics['lr'] = self.optimizer.state_dict()['param_groups'][0]['lr']
            else:
                step_metrics['lr'] = 0.0
        return step_losses, step_metrics

KerasModel.StepRunner = StepRunner

# 仅仅保存lora可训练参数
def save_ckpt(self, ckpt_path='checkpoint.pt', accelerator=None):
    unwrap_net = accelerator.unwrap_model(self.net)
    unwrap_net.save_pretrained(ckpt_path)


def load_ckpt(self, ckpt_path='checkpoint.pt'):
    self.net = self.net.from_pretrained(self.net, ckpt_path)
    self.from_scratch = False


KerasModel.save_ckpt = save_ckpt
KerasModel.load_ckpt = load_ckpt


keras_model = KerasModel(model,loss_fn = None,
        optimizer=torch.optim.AdamW(model.parameters(),lr=2e-6))
ckpt_path = 'finance_chatglm2'

keras_model.fit(train_data = dl_train,
                val_data = dl_val,
                epochs=100,patience=5,
                monitor='val_loss',mode='min',
                ckpt_path = ckpt_path,
                mixed_precision='fp16'
               )


# model = AutoModel.from_pretrained("D:/LLM_dev/FinGPT-intern/finetune/chatglm2-6b-finance",
#                                   load_in_8bit=False,
#                                   trust_remote_code=True,
#                                   device_map='auto')
# model = PeftModel.from_pretrained(model, ckpt_path)
# model = model.merge_and_unload()  # 合并lora权重
model.save_pretrained("chatglm2-6b-finance", max_shard_size='1GB')
tokenizer.save_pretrained("chatglm2-6b-finance")
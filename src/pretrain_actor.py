import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import lightning as L
from lightning.pytorch.loggers import WandbLogger

from tensordict import TensorDict

from pkg_ddpg_td3.utils.map import generate_map_static
from pkg_torchrl.base import build_actor
from pkg_torchrl.env import make_env
from pkg_torchrl.pretrain import SimpleController, rollout

from configs import BaseConfig


class BC(L.LightningModule):
    def __init__(self, train_env, config):
        super().__init__()
        self.lr = config.pretrain.lr

        in_keys = ['observation']
        action_spec = train_env.action_spec
        obs_size = 0
        for key in in_keys:
            obs_size += train_env.observation_spec[key].shape[-1]
        if train_env.batch_size:
            action_spec = action_spec[(0,) * len(train_env.batch_size)]

        self.actor = build_actor(obs_size, action_spec, in_keys, config.sac, use_random_interaction=False)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        td_x = TensorDict({'observation': x})
        td_out = self.actor(td_x)
        return td_out['action']

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        self.log('train_loss', loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        self.log('val_loss', loss, prog_bar=True)
        return {'val_loss': loss}
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


def load_data(env, config):
    ref_speed = .8

    model = SimpleController(ref_speed)
    data = rollout(env, model, config, n_steps=config.pretrain.rollout_steps)

    obs = torch.stack(data['obs'])
    action = torch.stack(data['action'])

    dataset = TensorDataset(obs, action)

    train_size = int(0.8 * len(dataset))
    eval_size = len(dataset) - train_size

    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])
    train_loader = DataLoader(train_dataset, batch_size=config.pretrain.bs, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=config.pretrain.bs, shuffle=False)

    return train_loader, eval_loader

def main():
    config = BaseConfig()
    env = make_env(config, generate_map=generate_map_static)

    # load data
    train_loader, eval_loader = load_data(env, config)

    # regress actor on reference trajectory data
    model = BC(env, config)

    wandb_logger = WandbLogger(project='DRL-Traj-Planner', tags=['pretrain'])
    trainer = L.Trainer(max_epochs=config.pretrain.epochs,
                        logger=wandb_logger,
                        gradient_clip_val=.5,
                        enable_checkpointing=False)
    trainer.fit(model, train_loader, eval_loader)

    # save actor
    torch.save(model.actor.state_dict(), "../Model/testing/pretrained_actor.pt")


if __name__ == '__main__':
    main()
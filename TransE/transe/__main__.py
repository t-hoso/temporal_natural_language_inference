import click
from torch._C import TensorType
from torch.utils import tensorboard

from .controller.train import Training


@click.command()
@click.option("--model_name", required=True, type=str)
@click.option("--mapped_embedding_dim", default=256, type=int)
@click.option("--relation", required=True, type=str, multiple=True)
@click.option("--tensorboard_path", required=True, type=str)#click.Path(exists=False))
@click.option("--lr", required=True, type=float)
@click.option("--n_epochs", default=30)
@click.option("--model_save_path", type=str)
@click.option("--margin", default=0.5)
@click.option("--epsilon", default=1e-10)
@click.option("--constraint_weight", default=0.25)
def run(model_name,
        mapped_embedding_dim,
        relation,
        tensorboard_path,
        lr,
        n_epochs,
        model_save_path, # r"D:\workspace\study\transe\model\30eps-isAfter-isBefore-TransH_constrained",
        margin, 
        epsilon, 
        constraint_weight):
    Training.train(model_name,
                   mapped_embedding_dim,
                   relation,
                   tensorboard_path,
                   lr,
                   n_epochs=n_epochs,
                   model_save_path=model_save_path,
                   margin=margin, 
                   epsilon=epsilon, 
                   constraint_weight=constraint_weight
                   )


run()
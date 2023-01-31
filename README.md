# VS-CNN-Children-Drawings

Este código é para o meu Trabalho de Conclusão de Curso para a universidade da PUC-RS do curso de Pós-Graduação em Ciência de Dados e Inteligência Artificial.

## Repositório

Neste repositório contém:

- Código para treinar e validar o modelo
- Os dados utilizados para a realização do treinamento
- [Os pesos (weights) de pré-treino](https://drive.google.com/file/d/1nXPiqc8uMxgd9dlIT0JooSoMEpMhXxCB/view?usp=share_link) para a inicialização do projeto

## Requisitos mínimos

- Python 3
- Tensorflow > 1.0
- Tqdm

## Treinamento e Validação

- Modelo Base:

```bash
python train_base.py --dataset [children]
```

- Modelo Factor:

Para treinar o modelo factor, é necessário ter os pesos de pré-treino dos modelos base para iniciação.

```bash
python train_factor.py --dataset [children] --factor_layer [conv1,conv3,conv5,fc7] --num_factors 16
```

## Referências

A realização deste TCC foi inspirado e altamente auxiliado com os repositórios abaixo:

@inproceedings{vs-cnn,
  title={Visual sentiment analysis for review images with item-oriented and user-oriented CNN},
  author={Truong, Quoc-Tuan and Lauw, Hady W},
  booktitle={Proceedings of the ACM on Multimedia Conference},
  year={2017},
}

@inproceedings{wang2019learning,
  title={Learning Robust Global Representations by Penalizing Local Predictive Power},
  author={Wang, Haohan and Ge, Songwei and Lipton, Zachary and Xing, Eric P},
  booktitle={Advances in Neural Information Processing Systems},
  pages={10506--10518},
  year={2019}
}
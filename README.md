# Schema-guided User Satisfaction Modeling for Task-oriented Dialogues

This repository contains the code for the paper titled **[Schema-guided User Satisfaction Modeling for Task-oriented Dialogues]()**, which is accepted by the ACL 2023.

## 1. Installation
```sh
conda create --name usm python=3.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```

## 2. Preprocess the data
The dataset can be downloaded via the following repository: [USDA](https://github.com/dengyang17/USDA/tree/main).

## 3. Training
```sh
./train.sh
```

## 4. Testing
```sh
./test.sh
```

## Bugs or questions?
If you have any inquiries pertaining to the code or the paper, please do not hesitate to contact [Yue Feng](https://yuefeng-leah.github.io/homepage/main). In case you encounter any issues while utilising the code or wish to report a bug, you may open an issue. We kindly request that you provide specific details regarding the problem so that we can offer prompt and efficient assistance.

## Citation
```
@inproceedings{yue2023usm,
  title={Schema-guided User Satisfaction Modeling for Task-oriented Dialogues},
  author={Yue Feng, Yunlong Jiao, Animesh Prasad, Nikolaos Aletras, Emine Yilmaz and Gabriella Kazai},
  year={2023},
  address = {Toronto, Canada},
  booktitle = {The 61st Annual Meeting of the Association for Computational Linguistics: ACL 2023},
  publisher = {Association for Computational Linguistics},
}
```

## Authors

- **Yue Feng**: Main contributor

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.

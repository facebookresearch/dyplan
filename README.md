# DyPlan

This is the code for our [paper](https://arxiv.org/pdf/2410.23511) "Dynamic Strategy Selection for Efficient Question Answering using Large Language Models" which has been accepted at NAACL Findings 2025.

---

These are the general steps to create training data and evaluate the DyPlan-trained models.

### QA Data

We benchmark our work on three prominent datasets of [HotpotQA](https://aclanthology.org/D18-1259/), [2WikiMultihopQA](https://aclanthology.org/2020.coling-main.580/), and [Musique](https://aclanthology.org/2022.tacl-1.31/). Refer to the original paper data releases to procure the data and put them in the `DRAGIN/data/<dataname>` folder.

### Setting up MHQA

We majorly refer to [DRAGIN](https://github.com/oneal2000/DRAGIN/tree/main) for setting up the Multi-hop Question-Answering (MHQA) base-LLM runs and evaluation. We make modifications to their files to adapt to our setting. Utilize the below script to pull DRAGIN's repo and move our modified files to the appropriate folders.

```
    bash setup_mhqa.sh
```

### Create base-LLM training data

Once you set up DRAGIN, the first step is to create base LLM outputs for each strategy of interest. For this, you need to create appropriate config files for the strategy. For example, we provide a reference config files for various strategies in the `DRAGIN/config` folder.

Using the config files, you can use the following command to generate the base LLM strategy data. (Ensure you are in the `DRAGIN` directory to run this command)
```
    bash run_base_llm.sh 1000 hotpotqa direct_0-4k Direct
```

This will create corresponding output folder with evaluation in the `result` folder.

### Create Intermediate Classifier data

We next post-process the data to create an intermediate classifier layer data. This data can be utilized for training classifiers or useful in creating the final SFT data for LLM fine-tuning as well. To create this data, run the following command from the `DRAGIN` directory (example shown for rag).
```
    python src/create_classifier_data.py --fields rag --outputs_file ./result/llama3_8b_chat_2wikimultihopqa/%s_0-4k/output.txt --details_file ./result/llama3_8b_chat_2wikimultihopqa/%s_0-4k/details.txt --gold_file ./data/2wikimultihopqa/train.json --save_dir ./result/llama3_8b_chat_2wikimultihopqa/rag_0-4k/binary_rag --is_binary 1 --include_reasoning 1
```

The above command can be modified using the dev data and with `dev_split 0` to create test data as well. This will create intermediate classifier data in the corresponding `save_dir`.

### Setting up SFT using LLaMa-Factory

We majorly utilize [LLaMa-Factory](https://github.com/hiyouga/LLaMA-Factory/tree/main) for fine-tuning our models and utilize their data format for our data creation. Utilize the below script to pull this repo and move additional files to the appropriate folders.

```
    bash setup_llama_factory.sh
```

### Create SFT Training Data

Once you setup LLaMa-Factory, to create SFT data, you first need to change directory to `LLaMa-Factory/lf_sft_scripts`. Now you can run the following command to create the end2end SFT data for DyPlan.
```
    python lf_sft_scripts/data_creation/create_e2e_data.py --fields direct selfask cot rag --data_dirs ../../DRAGIN/result/llama3_8b_chat_2wikimultihopqa/direct_0-4k/binary_direct ../../DRAGIN/result/classifier_expts/data/llama3_8b_chat_2wikimultihopqa/cot_0-4k/binary_cot ../../DRAGIN/result/classifier_expts/data/llama3_8b_chat_2wikimultihopqa/selfask_0-4k/binary_selfask ../../DRAGIN/result/classifier_expts/data/llama3_8b_chat_2wikimultihopqa/rag_0-4k/binary_rag --save_file ./data/mhqa/2wikimultihopqa_train-4k_llama3-8b_e2e-direct-selfask-cot-rag_best-all-fields_multi-turn.json --include_best_preplan 1 --num_samples 4000
```
You can use the `--verify 1` flag to create SFT data for Dyplan-verify version of the model. You can use the above script to appropriately create dev data as well. To create test data for Dyplan, you can utilize the `create_e2e_data_test_multiturn.py` script in a similar fashion. For Dyplan-verify, we don't need to create direct test data.

Make sure to add an entries for the created datasets in `LLaMa-Factory/data/dataset_info.json` file for internal model reference. (Refer to LLaMa-Factory for more details here).

### Training the model

To train the LLM, you first need to create an appropriate config file. We have added one for reference. Next, you can follow standard procedure of LLaMa-Factory by using this config file from the `LLaMa-Factory` directory as
```
    llamafactory-cli train ./lf_sft_scripts/config/llama3-8b_lora-r32_sft_2wikimultihopqa_e2e-direct-selfask-cot-rag_best-all-fields_multi-turn-mask_4k.yaml
```

### Evaluating the model

For evaluation, we run the model for dev-set evaluation on all checkpoints and then run on test set for best checkpoint. For DyPlan evaluation, refer to the `LLaMa-Factory/lf_sft_scripts/scripts/evalall_lora_sft_multi_turn.sh` script. For DyPlan-verify evaluation, refer to the `LLaMa-Factory/lf_sft_scripts/scripts/evalall_lora_sft_multiturn_verify.sh` script. You can modify these scripts as needed, but these are the main evaluation files.

Once the `generated_predictions.jsonl` is create for checkpoints, we can utilize the DRAGIN evaluation framework for fetching the EM/F1 performance numbers along with the #T/#R cost numbers. Majorly you can use the `DRAGIN/src/evaluate_sft.py` file for each of these checkpoints for this evaluation.

## License

Majority of our code and models are licensed under [CC BY-NC 4.0](https://github.com/facebookresearch/dyplan/blob/main/LICENSE).
Some portions of the project are modified from [DRAGIN](https://github.com/oneal2000/DRAGIN/tree/main) and licensed separately under their [original license](https://github.com/oneal2000/DRAGIN/blob/main/LICENSE).
Some portions of the project are modified from [LLaMa-Factory](https://github.com/hiyouga/LLaMA-Factory/tree/main) and licensed separately under their [original license](https://github.com/hiyouga/LLaMA-Factory/blob/main/LICENSE).


## Contributing

See [contributing](https://github.com/facebookresearch/dyplan/blob/main/CONTRIBUTING.md) and the [code of conduct](https://github.com/facebookresearch/dyplan/blob/main/CODE_OF_CONDUCT.md).

## Citation

If you find our work useful and would like to cite us, please use the citation bibtex below
```
@article{parekh2024dynamic,
  title={Dynamic Strategy Planning for Efficient Question Answering with Large Language Models},
  author={Parekh, Tanmay and Prakash, Pradyot and Radovic, Alexander and Shekher, Akshay and Savenkov, Denis},
  journal={Proceedings of the 2025 Conference of the North American Chapter of the Association for Computational Linguistics (NAACL) Findings},
  year={2025}
}
```

## Contact

For any issues, please reach out to Pradyot Prakash (pradyot@meta.com).

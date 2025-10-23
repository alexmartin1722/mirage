# MiRAGE: Multimodal Retrieval-Augmented Generation Evaluation

<div align="center">
<a href="" target="_blank"><img src=https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv></a>
<a href="" target="_blank"><img src=https://img.shields.io/badge/HuggingFace-Evaluate-FF6D00?logo=huggingface></a>
<a href="https://opensource.org/license/apache-2-0"><img src="https://img.shields.io/static/v1?label=License&message=Apache-2.0&color=red"></a>
</div>

MiRAGE: Multimodal Retrieval-Augmented Generation Evaluation. 

## Contents
* [Features](#features)
* [Supported Tasks](#supported-tasks)
* [Installation](#installation)
* [MiRAGE Usage](#mirage-usage)
* [Other RAG Metric Usage](#other-rag-metric-usage-only-videorag-supported)
* [Citation](#citation)
* [Contacts](#contacts)


## Features
- Evaluating multimodal retrieval-augmented generation systems.
- Integration with vLLM, DeepSpeed, FlashAttention, and other efficient inference techniques.
- Easy-to-use command line interface for running various metrics.
- Evaluation for generation from videos.

### Coming Soon
- Support for calibrated models for claim verification in videos.
- Support for more modalities (text, image, audio).

## Supported Tasks
### Video RAG
- WikiVideo: [repo](https://github.com/alexmartin1722/wikivideo), [paper](https://arxiv.org/abs/2504.00939)

### Text RAG 
- NeuCLIR: [repo](), [paper]()

### Visual Document RAG 
- Coming soon

### Audio RAG 
- Coming less soon

## Installation
<details><summary><b>From PyPI</b></summary>

Coming soon...

```bash
pip install mirage-eval
```
</details>

<details><summary><b>From Scratch</b></summary>

```bash
conda create -n video_rag_eval python=3.12 -y 
conda activate video_rag_eval
pip install --upgrade uv
uv pip install vllm --torch-backend=cu128
pip install evaluate 
pip install qwen-vl-utils[decord]==0.0.8
pip install peft
```
</details>

<details><summary><b>From Environment File</b></summary>

```bash
```

</details>

## MiRAGE Usage

### VideoRAG Evaluation
<details><summary><b>Data Prep</b></summary>

When evaluating VideoRAG, you will need the following data:

- predictions, 
- references, 
- video directory, containing all the videos possible to use in RAG (for collection eval only),

#### WikiVideo Data 
We provide everything need to evaluate on 


#### Custom data 

</details>

<details><summary><b>Evaluation</b></summary>
When evaluating 

#### InfoF1:
```bash
python infof1.py \
    --eval_type [reference|collection] \
    --prediction [path_to_system_prediction] \
    --reference [path_to_human_eval_json] \
    --video_dir [path_to_videos] \ #only needed for collection eval
    --output_dir [path_to_output_directory] \
    --model_name [qwen_7b|qwen_72b]
```
```bash
python infof1.py \
    --eval_type collection \
    --prediction data/wikivideo/model_preds/qwen_72b_cag_relevant_citations.json \
    --reference data/wikivideo/human_eval_subset.json \
    --video_dir /exp/amartin/wikivideo/all_videos \
    --output_dir data/wikivideo/model_preds/metric_outputs \
    --model_name qwen_7b
```
#### CiteF1:
```bash
python citef1.py \
    --eval_type [reference|collection] \
    --prediction [path_to_system_prediction] \
    --reference [path_to_human_eval_json] \
    --video_dir [path_to_videos] \ #only needed for collection eval
    --output_dir [path_to_output_directory] \
    --model_name [qwen_7b|qwen_72b]
```
```bash
python citef1.py \
    --eval_type collection \
    --prediction data/wikivideo/model_preds/qwen_72b_cag_relevant_citations.json \
    --reference data/wikivideo/human_eval_subset.json \
    --video_dir /exp/amartin/wikivideo/all_videos \
    --output_dir data/wikivideo/model_preds/metric_outputs \
    --model_name qwen_7b
```

</details>


### TextRAG Evaluation (Coming Soon)
<details><summary><b>Data Prep</b></summary>


</details>

<details><summary><b>Evaluation</b></summary>


</details>



## Other RAG Metric Usage (Only VideoRAG Supported)
### ALCE Evaluation 


### RAGAS Evaluation


### ARGUE Evaluation
<details><summary><b>Data Prep</b></summary>

</details>

<details><summary><b>Evaluation</b></summary>
Coming Soon. This was really messy to implement. For now we recommend 
</details>





## Citation
If you find MiRAGE useful in your research, please consider citing the following paper:

```
```

### Citing calibrated models
Coming soon

### Citing the other RAG Metrics
#### ALCE
```bibtex
@inproceedings{gao-etal-2023-enabling,
    title = "Enabling Large Language Models to Generate Text with Citations",
    author = "Gao, Tianyu  and
      Yen, Howard  and
      Yu, Jiatong  and
      Chen, Danqi",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.398/",
    doi = "10.18653/v1/2023.emnlp-main.398",
    pages = "6465--6488"
}
```
#### ARGUE
```bibtex
@misc{walden2025autoarguellmbasedreportgeneration,
      title={Auto-ARGUE: LLM-Based Report Generation Evaluation}, 
      author={William Walden and Orion Weller and Laura Dietz and Bryan Li and Gabrielle Kaili-May Liu and Yu Hou and Eugene Yang},
      year={2025},
      eprint={2509.26184},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2509.26184}, 
}
```
```bibtex
@inproceedings{Mayfield_2024, series={SIGIR 2024},
   title={On the Evaluation of Machine-Generated Reports},
   url={http://dx.doi.org/10.1145/3626772.3657846},
   DOI={10.1145/3626772.3657846},
   booktitle={Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval},
   publisher={ACM},
   author={Mayfield, James and Yang, Eugene and Lawrie, Dawn and MacAvaney, Sean and McNamee, Paul and Oard, Douglas W. and Soldaini, Luca and Soboroff, Ian and Weller, Orion and Kayi, Efsun and Sanders, Kate and Mason, Marc and Hibbler, Noah},
   year={2024},
   month=jul, pages={1904–1915},
   collection={SIGIR 2024} }
```

#### RAGAS
```bibtex
@inproceedings{es-etal-2024-ragas,
    title = "{RAGA}s: Automated Evaluation of Retrieval Augmented Generation",
    author = "Es, Shahul  and
      James, Jithin  and
      Espinosa Anke, Luis  and
      Schockaert, Steven",
    editor = "Aletras, Nikolaos  and
      De Clercq, Orphee",
    booktitle = "Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics: System Demonstrations",
    month = mar,
    year = "2024",
    address = "St. Julians, Malta",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.eacl-demo.16/",
    doi = "10.18653/v1/2024.eacl-demo.16",
    pages = "150--158",
}
```


## Contacts
If you have MiRAGE specific questions, would like a new feature, model support, supported dataset, etc., feel free to open an issue. 

You can also reach out to me for general comments/suggestions/questions through email. 
- Alexander Martin, amart233@jhu.edu
    - if the email listed there is out of date, you can find my current email on my [personal website](https://alexmartin1722.github.io/).

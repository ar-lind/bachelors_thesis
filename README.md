# Improving Automatic Speech Recognition for Low-Resource Languages 
***
In this file, you will find example instructions for running the experiments presented in our thesis 
(Improving Automatic Speech Recognition for Low-Resource Languages).
***
## Transcription experiments:

### Comparison between before and after training:

#### XLS-R:
Here you can find the files related to 
the ASR results before and after training the XLS-R model.

To avoid an out of bounds error, you have to replace the 
[original ctc_prefix_score.py](https://github.com/juice500ml/espnet/blob/wav2gloss/espnet/nets/ctc_prefix_score.py)
with the
[ctc_prefix_score.py](transcription_experiments/comparison_between_before_and_after_training/xls-r/ctc_prefix_score.py)
provided in this repository. 

You can copy the 
[run_before_1_xls-r.sh](transcription_experiments/comparison_between_before_and_after_training/xls-r/run_before_1_xls-r.sh) 
file and run it in the 
[original Wav2Gloss repository](https://github.com/juice500ml/espnet/tree/wav2gloss/egs2/wav2gloss/asr1).
This will create the Beja checkpoint for the before experiment.

Then you can copy the 
[run_before_2_xls-r.sh](transcription_experiments/comparison_between_before_and_after_training/xls-r/run_before_2_xls-r.sh) 
file and run it in the 
[original Wav2Gloss repository](https://github.com/juice500ml/espnet/tree/wav2gloss/egs2/wav2gloss/asr1).
This will compute the "before" results for the Savosavo language.

Then you can copy the 
[run_after_xls-r.sh](transcription_experiments/comparison_between_before_and_after_training/xls-r/run_after_xls-r.sh) 
file and run it in the 
[original Wav2Gloss repository](https://github.com/juice500ml/espnet/tree/wav2gloss/egs2/wav2gloss/asr1).
This will compute the "after" results for the Savosavo language.

To compute the results for other languages, you can change the language tag `lang="savo1255"` on line 10 to the target
language's tag.




#### OWSM:
Here you can find the files related to 
the ASR results before and after training the OWSM model.

To avoid errors, we changed the [original OWSM Wav2Gloss repository](https://github.com/juice500ml/finetune_owsm)'s
[dataset.py](transcription_experiments/comparison_between_before_and_after_training/owsm/dataset.py) and 
[finetune.py](transcription_experiments/comparison_between_before_and_after_training/owsm/finetune.py).
In the [dataset.py](transcription_experiments/comparison_between_before_and_after_training/owsm/dataset.py) file, we
changed line 94 to 
```python
self.save_hyperparameters(ignore=["new_tokens"])
```
and in the [finetune.py](transcription_experiments/comparison_between_before_and_after_training/owsm/finetune.py) file, 
we changed line 129 to 
```python
def test_step(self, batch, batch_idx, dataloader_idx=0):
```

You can replace the original 
[dataset.py](https://github.com/juice500ml/finetune_owsm/dataset.py) and 
[finetune.py](https://github.com/juice500ml/finetune_owsm/finetune.py) files and run 
```
python3 finetune.py --tasks transcription --langs beja1238 --batch_size 16 --exp_name transcription_beja1238
```
in the [original OWSM Wav2Gloss repository](https://github.com/juice500ml/finetune_owsm).
This will create the Beja checkpoint.

Then you can run
```
python3 evaluate_model.py --langs "savo1255" --tasks "transcription" --devices 1 --checkpoint_path "exps/transcription_beja1238..."
```
with the according generated checkpoint in the exps directory as checkpoint_path in the 
[original OWSM Wav2Gloss repository](https://github.com/juice500ml/finetune_owsm).
This will compute the "before" results for the Savosavo language.

To compute the results for other languages, you can change the language argument `--langs savo1255` to the target 
language's tag.

You can run 
```
python3 finetune.py --tasks transcription --langs savo1255 --batch_size 16 --exp_name transcription_savo1255
```
and
```
python3 evaluate_model.py --langs "savo1255" --tasks "transcription" --devices 1 --checkpoint_path "exps/transcription_savo1255..."
```
to compute the "after" results for the Savosavo language.

To compute the results for other languages, you can change the language arguments `--langs savo1255` to the target 
language's tags.

### Transcription of unseen languages:

#### XLS-R:
Again, if not already done, to avoid the out of bounds error, you have to replace the 
[ctc_prefix_score.py](https://github.com/juice500ml/espnet/blob/wav2gloss/espnet/nets/ctc_prefix_score.py)
with the
[ctc_prefix_score.py](transcription_experiments/comparison_between_before_and_after_training/xls-r/ctc_prefix_score.py)
provided in this repository. 

You can repeat the "Comparison between before and after training"-before runs with Kalamang, using the files 
[run_before_1_xls-r.sh](transcription_experiments/transcription_of_unseen_languages/run_before_1_xls-r.sh),
[run_before_2_xls-r.sh](transcription_experiments/transcription_of_unseen_languages/run_before_2_xls-r.sh),
and [run_after_xls-r.sh](transcription_experiments/transcription_of_unseen_languages/run_after_xls-r.sh).

#### OWSM:
You can replace the original 
[dataset.py](https://github.com/juice500ml/finetune_owsm/dataset.py) and 
[finetune.py](https://github.com/juice500ml/finetune_owsm/finetune.py) files again and run 
```
python3 finetune.py --tasks transcription --langs kara1499 --batch_size 16 --exp_name transcription_kara1499
```
in the [original OWSM Wav2Gloss repository](https://github.com/juice500ml/finetune_owsm).

Then you can repeat the "Comparison between before and after training"-before runs with Kalamang, using 
```
python3 evaluate_model.py --langs "kara1499" --tasks "transcription" --devices 1 --checkpoint_path "exps/transcription_beja1238..."
```
with the corresponding generated Beja checkpoint in the exps directory as checkpoint_path in the 
[original OWSM Wav2Gloss repository](https://github.com/juice500ml/finetune_owsm).
This will compute the results for the unseen Kalamang language.

### Transcription of seen languages:

#### XLS-R:
Again, if not already done, to avoid the out of bounds error, you have to replace the 
[ctc_prefix_score.py](https://github.com/juice500ml/espnet/blob/wav2gloss/espnet/nets/ctc_prefix_score.py)
with the
[ctc_prefix_score.py](transcription_experiments/comparison_between_before_and_after_training/xls-r/ctc_prefix_score.py)
provided in this repository. 

You can copy the 
[run_xls-r.sh](transcription_experiments/transcription_of_seen_languages/run_xls-r.sh) file and run it in the 
[original Wav2Gloss repository](https://github.com/juice500ml/espnet/tree/wav2gloss/egs2/wav2gloss/asr1).
This will compute the results for the seen Beja language.

To compute the results for other languages, you can change the language tag `lang="beja1238"` on line 10 to the target
language's tag.

#### OWSM:
You can replace the original 
[dataset.py](https://github.com/juice500ml/finetune_owsm/dataset.py) and 
[finetune.py](https://github.com/juice500ml/finetune_owsm/finetune.py) files again and run 
```
python3 finetune.py --tasks transcription --langs beja1238 --batch_size 16 --exp_name transcription_beja1238
```
in the [original OWSM Wav2Gloss repository](https://github.com/juice500ml/finetune_owsm).

Then you can run
```
python3 evaluate_model.py --langs "beja1238" --tasks "transcription" --devices 1 --checkpoint_path "exps/transcription_1238..."
```
with the corresponding generated checkpoint in the exps directory as checkpoint_path in the 
[original OWSM Wav2Gloss repository](https://github.com/juice500ml/finetune_owsm).
This will compute the results for the seen Beja language.

To compute the results for other languages, you can change the language argument `--langs beja1238` to the target 
language's tag.


***


## Post-processing simulated noisy transcriptions:
For the post-processing experiments we used training arguments with reduced saving, due to limited space. Either use 
these arguments and check the logged scores during the training for the best ones or adjust the arguments, e.g.,
`save_strategy`, `save_total_limit`, and `load_best_model_at_end`.


### Amount of noise:
Run [amount_noise.py](post-processing_simulated_noisy_transcriptions/amount_of_noise/amount_noise.py) to compute the
base results, change the `add_base_noise` on line 26 in 
[dataset_amount_noise.py](post-processing_simulated_noisy_transcriptions/amount_of_noise/dataset_amount_noise.py) to 
`add_high_noise` to compute the high-noise results, or change to 
`add_low_noise` to compute the low-noise results.


### Dataset structure:
Run [structure.py](post-processing_simulated_noisy_transcriptions/dataset_structure/structure.py) to compute the base 
structure result. If you want to compute the single- or double-noise structure results, remove the comments of the
corresponding section in the 
[dataset_structure.py](post-processing_simulated_noisy_transcriptions/dataset_structure/dataset_structure.py)
and comment out the others.


### Denoising model size:
Run [model_size.py](post-processing_simulated_noisy_transcriptions/model_size/model_size.py) to compute the base model 
size's results, or change `google/byt5-base` to `google/byt5-small` or `google/byt5-large` in the
[config.yaml](post-processing_simulated_noisy_transcriptions/model_size/configs/config.yaml)
to compute the other model
sizes' results.


### Seen languages:
Run [seen_and_unseen.py](post-processing_simulated_noisy_transcriptions/seen_and_unseen_languages/seen_and_unseen.py) to
compute the results from the seen languages experiments. To compute other languages' results, change the language tag
`language: "beja1238"` in the 
[config.yaml](post-processing_simulated_noisy_transcriptions/seen_and_unseen_languages/configs/config.yaml) to 
`"even1259"`, `"apah1238"`, or `"savo1255"`.

### Unseen languages:
To compute the unseen Kalamang results, change line 153 in the 
[seen_and_unseen.py](post-processing_simulated_noisy_transcriptions/seen_and_unseen_languages/seen_and_unseen.py) file 
to 
```python
training(mode="unseen")
```
before running.
You also need to add a folder containing the Kalamang examples from the
[IMT Vault](https://imtvault.org/?languageNameFilter=kalaman&languageName[0]=Kalamang&p=4).

### Real ASR evaluation after simulated noisy training:

#### OWSM:
Run [owsm_sim_train.py](post-processing_simulated_noisy_transcriptions/real_ASR_evaluation_after_simulated_noisy_training/owsm/owsm_sim_train.py)
to compute the results from the experiment, where we trained the model on simulated noisy text and evaluated it on
real predictions of the OWSM model.
To compute Evenki results, change the language tag
`language: "beja1238"` in the 
[config.yaml](post-processing_simulated_noisy_transcriptions/real_ASR_evaluation_after_simulated_noisy_training/owsm/configs/config.yaml) 
to `"even1259"`

#### XLS-R:
Run [xls-r_sim_train.py](post-processing_simulated_noisy_transcriptions/real_ASR_evaluation_after_simulated_noisy_training/xls-r/xls-r_sim_train.py)
to compute the results from the experiment, where we trained the model on simulated noisy text and evaluated it on
real predictions of the XLS-R model.
To compute Evenki results, change the language tag
`language: "beja1238"` in the 
[config.yaml](post-processing_simulated_noisy_transcriptions/real_ASR_evaluation_after_simulated_noisy_training/xls-r/configs/config.yaml) 
to `"even1259"`
***


## Post-processing real ASR transcriptions:
For the post-processing experiments we used training arguments with reduced saving, due to limited space. Either use 
these arguments and check the logged scores during the training for the best ones or adjust the arguments, e.g.,
`save_strategy`, `save_total_limit`, and `load_best_model_at_end`.

### Training on a single language:
Run [single.py](post-processing_real_ASR_transcriptions/training_on_a_single_language/single.py) to compute the results
from the experiment, where we did single-language training and evaluation for post-processing real transcription 
predictions. To compute the results for other languages, change the language tag `language: "beja1238"` in the 
[config.yaml](post-processing_real_ASR_transcriptions/training_on_a_single_language/configs/config.yaml)
to `"even1259"`, `"selk1253"`, `komn1238`, or `goro1270`.

### Training on multiple languages:
Run [multi_tags.py](post-processing_real_ASR_transcriptions/training_on_multiple_languages/multi_tags.py) to compute the results
from the experiment, where we did multi-language training and evaluation for post-processing real transcription 
predictions with language tags. To change the evaluation language, adjust the language tag 
`language: "beja1238"` in the 
[config.yaml](post-processing_real_ASR_transcriptions/training_on_multiple_languages/configs/config.yaml) to 
`"even1259"`, `"selk1253"`, `komn1238`, or `goro1270`.

### Usage of language tags:
Run [multi_no_tags.py](post-processing_real_ASR_transcriptions/usage_of_language_tags/multi_no_tags.py) to compute the 
Beja result from the experiment, where we did multi-language training and evaluation for post-processing real 
transcription predictions without the use of language tags.

***


## IPA generation:

### Generating audio-based IPA transcriptions:
To compute the results from the audio-based IPA transcription experiments, run 
[audio-based_ipa.py](IPA_generation/generating_audio-based_IPA_transcriptions/audio-based_ipa.py).
To compute the results for other languages, change the language tag `language: "beja1238"` in the 
[config.yaml](IPA_generation/generating_audio-based_IPA_transcriptions/configs/config.yaml) to 
`"even1259"`, or `"selk1253"`.

### Generating text-based IPA transcriptions:
To compute the result from the text-based IPA transcription experiment, run 
[text-based_ipa.py](IPA_generation/generating_text-based_IPA_transcriptions/text-based_ipa.py).

***


## Large Language Model experiments:

### Google Gemini 2.5 Flash Lite:
Run [gemini.py](Large_Language_Model_experiments/google_gemini/gemini.py) to compute the proprietary LLM results.
To access the model, add your api key to the 
[config.yaml](Large_Language_Model_experiments/google_gemini/configs/config.yaml).
You can also find a selection of prompts in the comments in 
[gemini.py](Large_Language_Model_experiments/google_gemini/gemini.py).

### Qwen3 Omni 30B-A3B-Instruct:
 To access the model, follow the instructions in the 
[original Owen3 Omni 30B-A3B-Instruct Page](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) to download the 
model.

Run [qwen3_omni.py](Large_Language_Model_experiments/qwen3_omni/qwen3_omni.py) to compute the small and open-weight 
multimodal LLM results. You need to create a folder with the fieldwork data as described in the comments. To compute 
results for Farsi, change
```python
LANG_ID = "apah1238"
LANG_FULL = "Yali (Apahapsali)"
```
to 
```python
LANG_ID = "tehr1242"
LANG_FULL = "Farsi"
```
.

***


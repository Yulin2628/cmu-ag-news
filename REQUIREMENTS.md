# Assignment coverage

Based on the supplied CMU 11-667 Fall 2025 mini-project specification.

| Requirement | Implementation and evidence |
| --- | --- |
| Task, motivation, data, ethics | Report sections 1-2; split manifest and text hashes |
| At least 1,500 dataset examples | 2,000 train + 400 development + 400 fresh test |
| Input/target formulation and scoring | Answer-only training; generated-label exact matching; report section 2 |
| ICL for two models | Independent shot-count, class-definition, explicit-instruction and chat-turn searches |
| Fine-tuning for two models | Two LoRA learning rates per model, selected on development accuracy |
| Required comparisons and overfitting | Four systems on the same fresh test; train/dev/test comparisons |
| Quantitative and qualitative errors | Per-class scores, confusion matrices, paired tests and actual generated outputs |
| Deployment choice | Report section 7 with class performance, throughput caveats and serving limitations |
| Additional experiment | Problem 4 option 5: matched fixed versus retrieved demonstrations |
| Reproducibility | Fixed seeds, complete code, protocol, predictions, environment and revision records |
| Presentation materials | Seven-slide editable deck covering task, metric, results and main finding |

These files are project materials. No course submission or classroom presentation is performed by publishing the repository.

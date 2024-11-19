#python -m chrisdata.cli ner compare_eval GNER/output/flan-t5-base-task-adaptation-12ep.out \
#                                         GNER/output/flan-t5-base-concept-learning-24ep.out \
#                                         GNER/output/flan-t5-base-comparison.xlsx

python -m chrisdata.cli ner compare_eval GNER/output/flan-t5-base-task-adaptation-12ep.out \
                                         GNER/output/llama3_1b_supervised-plus.out \
                                         GNER/output/llama3_1b_supervised.xlsx

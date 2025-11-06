## Usage

torch-model-archiver \
  --model-name aesthetic_v2_5 \
  --version 1.0 \
  --serialized-file assets/aesthetic_predictor_v2_5.pth \
  --handler handler.py \
  --extra-files model_definition.py \
  --requirements-file requirements.txt \
  --export-path model_store \
  -f

torchserve --start --ts-config config.properties

curl -X POST http://127.0.0.1:8080/predictions/aesthetic_v2_5 -T test.jpeg
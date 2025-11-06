# handler.py
import os
import io
import torch
import base64
from PIL import Image
from ts.torch_handler.base_handler import BaseHandler

# Importa las definiciones de tu modelo personalizado
from model_definition import AestheticPredictorV2_5Model, AestheticPredictorV2_5Processor

class AestheticHandler(BaseHandler):
    """
    Handler personalizado para el modelo AestheticPredictorV2_5
    """
    def __init__(self):
        super().__init__()
        self.initialized = False

    def initialize(self, context):
        """
        Carga el modelo base (SigLIP) y aplica los pesos del head (.pth)
        """
        properties = context.system_properties
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")
        
        # Ruta al directorio del modelo, donde están los archivos del .mar
        model_dir = properties.get("model_dir")
        
        # 1. Cargar el modelo base SigLIP (el encoder)
        encoder_model_name = "google/siglip-so400m-patch14-384"
        print(f"Loading base model {encoder_model_name}...")
        self.model = AestheticPredictorV2_5Model.from_pretrained(encoder_model_name, torch_dtype="auto")
        
        # 2. Cargar el procesador
        self.processor = AestheticPredictorV2_5Processor.from_pretrained(encoder_model_name)
        
        # 3. Localizar y cargar los pesos del head (.pth)
        # El "serializedFile" es el .pth que pasamos al archiver
        serialized_file = context.manifest["model"]["serializedFile"]
        head_weights_path = os.path.join(model_dir, serialized_file)
        
        if not os.path.exists(head_weights_path):
            raise RuntimeError(f"Missing model weights file: {head_weights_path}")
            
        print(f"Loading custom head weights from {head_weights_path}...")
        state_dict = torch.load(head_weights_path, map_location="cpu")
        self.model.layers.load_state_dict(state_dict)
        
        self.model.to(self.device)
        self.model.eval()
        self.initialized = True
        print("Model loaded successfully.")

    def preprocess(self, data):
        """
        Preprocesa los datos de entrada (bytes de imagen)
        """
        images = []
        for req in data:
            # Obtener los bytes de la imagen
            image_bytes = req.get("data") or req.get("body")
            
            # Si es base64 (común en JSON), decodificar
            if isinstance(image_bytes, str):
                image_bytes = base64.b64decode(image_bytes)

            try:
                image = Image.open(io.BytesIO(image_bytes))
                if image.mode != "RGB":
                    image = image.convert("RGB")
                images.append(image)
            except Exception as e:
                raise ValueError(f"Failed to process input image: {e}")

        if not images:
            raise ValueError("No images found in request")
            
        # Procesar en batch
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        return inputs

    def inference(self, data, *args, **kwargs):
        """
        Ejecuta la inferencia
        """
        with torch.no_grad():
            outputs = self.model(**data)
        return outputs

    def postprocess(self, data):
        """
        Formatea la salida a un JSON
        """
        # data es el objeto ImageClassifierOutputWithNoAttention
        scores = data.logits
        scores_list = scores.cpu().tolist()
        
        # Formatear la salida: una lista de scores, uno por imagen
        results = [{"aesthetic_score": round(score[0], 4)} for score in scores_list]
        return results
from google.cloud import aiplatform, storage

# --- 1. Configuración ---
PROJECT_ID = "tu-proyecto-gcp"
REGION = "us-central1"
BUCKET_NAME = "tu-bucket-de-mlops" # El bucket donde está tu .mar
LOCAL_MODEL_PATH = "model-store/aesthetic_v2_5.mar"
GCS_MODEL_DIR = f"gs://{BUCKET_NAME}/models/" # Directorio en GCS
GCS_MODEL_FILE = f"{GCS_MODEL_DIR}aesthetic_v2_5.mar"

# --- 2. Sube tu .mar a GCS ---
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)
blob = bucket.blob("models/aesthetic_v2_5.mar")
blob.upload_from_filename(LOCAL_MODEL_PATH)
print(f"Modelo subido a {GCS_MODEL_FILE}")

# --- 3. Usa el contenedor de TORCHSERVE ---
# Esta es la imagen correcta para PyTorch 1.13 y CPU
# Puedes encontrar otras versiones aquí: 
# https://cloud.google.com/vertex-ai/docs/predictions/pre-built-containers
SERVING_CONTAINER_URI = "us-docker.pkg.dev/vertex-ai/prediction/torch-serve-cpu.1-13:latest"

aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET_NAME)

# --- 4. Registra el modelo ---
model = aiplatform.Model.upload(
    display_name="aesthetic-predictor-v2-5",
    
    # artifact_uri debe ser el DIRECTORIO que contiene el .mar
    artifact_uri=GCS_MODEL_DIR,
    
    # La imagen de contenedor CORRECTA
    serving_container_image_uri=SERVING_CONTAINER_URI,
    
    # Indica al contenedor de TorchServe qué modelo cargar
    # Vertex AI monta el 'artifact_uri' en /gcs/
    # (Esto puede variar, consulta la documentación si falla)
    #
    # Corrección: Google ahora prefiere que el .mar se llame 'model.mar'
    # o que no se especifiquen args, pero si tu .mar tiene otro nombre,
    # el método más seguro es no pasar 'serving_container_args'
    # y simplemente dejar que Vertex detecte 'aesthetic_v2_5.mar'.
    # 
    # Si tu handler.py espera el formato de bytes de Vertex:
    serving_container_predict_route="/predictions/aesthetic_v2_5",
    serving_container_health_route="/ping"
)

print(f"Modelo registrado: {model.resource_name}")

# --- 5. Despliega en un Endpoint ---
endpoint = model.deploy(
    machine_type="n1-standard-2", # Elige una máquina
    min_replica_count=1,
    max_replica_count=1
)

print(f"Endpoint desplegado: {endpoint.resource_name}")
print(f"URL de predicción: {endpoint.predict_endpoint}")
import cloudinary
import cloudinary.uploader
import cloudinary.api

# Cloudinary API
CLOUD_NAME='dntqqcuci'
API_KEY='182168662843965'
API_SECRET='3M7PhAHGCMOOr5EcImPt1g-bxvw'

# Cloudinary CONFIG
cloudinary.config(
    cloud_name=CLOUD_NAME,
    api_key=API_KEY,
    api_secret=API_SECRET
)

def uploadFile(fileAudio):
    cloudinary.config()
    upload_result = None
    file_to_upload = fileAudio

    if file_to_upload:
        upload_result = cloudinary.uploader.upload(file_to_upload, resource_type="video")
        return upload_result['secure_url']
        
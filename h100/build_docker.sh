docker build -t longtou/LaughterSegmentation:laughter .
docker tag longtou/LaughterSegmentation:laughter us-central1-docker.pkg.dev/prod-ai-project/tts/LaughterSegmentation:laughter

#docker run -it --runtime=nvidia longtou/LaughterSegmentation:laughter /bin/bash
#gcloud auth print-access-token | docker login -u oauth2accesstoken --password-stdin https://us-central1-docker.pkg.dev
#docker push us-central1-docker.pkg.dev/prod-ai-project/tts/LaughterSegmentation:laughter

# Stable Diffusion Image Generator App

 - First, run `download_sd_model.py` which will download and cache the model in an object store
```python
python download_sd_model.py
```
 - Next, grab the model URL and save is in the `MODEL_ENV` environment variable
```
export MODEL_DIR=s3://...
```
 - Finally, deploy the app in `image_generator_app.py`
```
python image_generator_app.py
```
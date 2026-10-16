[app]
title = Agni Eye
package.name = agnieye
package.domain = org.agnieye
source.include_exts = py,png,jpg,kv,atlas,tflite
source.include_dir = .
version = 1.0
requirements = python3,kivy,opencv-python-headless,numpy,tflite-runtime
orientation = portrait
fullscreen = 0
android.permissions = CAMERA,WRITE_EXTERNAL_STORAGE,READ_EXTERNAL_STORAGE
android.api = 33
android.minapi = 21
android.ndk = 25b
android.sdk = 30
android.archs = arm64-v8a

[buildozer]
log_level = 2
warn_on_root = 1

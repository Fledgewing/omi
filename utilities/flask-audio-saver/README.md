# README

## Purpose

Really simple container that looks like Whisper endpoint but just saves the wav files and returns nothing.  Useful to connect
Omi device to your own workflow entirely!

## Instructions

Publish this as follows:

```
docker build -t temporalise/flask-audio-saver .
docker push temporalise/flask-audio-saver
```

Run it locally with:
```
docker run -d -p 6000:5000 -v ./exports:/exports --name audio-saver temporalise/flask-audio-saver
```
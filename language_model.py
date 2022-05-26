import clip
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model, preprocess = clip.load("ViT-B/32")
model.to(device).eval()
input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size

def clip_encode_text(text_str):
    tokenized_text = clip.tokenize(text_str).cuda()
    text_features = model.encode_text(tokenized_text)
    return(text_features)

def clip_encode_images(image_input):
    resized_i = F.resize(image_input, size=(224, 224))
    image_features = model.encode_image(resized_i)
    return(image_features)

def get_clip_loss(text_str, images_tensor):
    image_features=clip_encode_images(images_tensor.cuda())
    text_features=clip_encode_text(text_str).cuda()
    text_features=text_features.repeat(image_features.shape[0], 1)
    similarity = text_features @ image_features.T
    return(-torch.mean(similarity))
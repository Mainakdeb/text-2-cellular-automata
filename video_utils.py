import numpy as np
from tqdm import tqdm_notebook, tnrange
import imageio
from base64 import b64encode
import moviepy.editor as mvp
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
import torch

class VideoWriter:
  def __init__(self, filename='./_autoplay.mov', fps=60.0, **kw):
    self.writer = None
    self.params = dict(filename=filename, fps=fps, **kw)

  def add(self, img):
    img = np.asarray(img)
    if self.writer is None:
      h, w = img.shape[:2]
      self.writer = FFMPEG_VideoWriter(size=(w, h), **self.params)
    if img.dtype in [np.float32, np.float64]:
      img = np.uint8(img.clip(0, 1)*255)
    if len(img.shape) == 2:
      img = np.repeat(img[..., None], 3, -1)
    self.writer.write_frame(img)

  def close(self):
    if self.writer:
      self.writer.close()

  def __enter__(self):
    return self

  def __exit__(self, *kw):
    self.close()
    if self.params['filename'] == '_autoplay.mov':
      self.show()

  def show(self, **kw):
      self.close()
      fn = self.params['filename']
      display(mvp.ipython_display(fn, **kw))

def zoom(img, scale=4):
   img = np.repeat(img, scale, 0)
   img = np.repeat(img, scale, 1)
   return img

def create_inference_video(ca_model, size, num_frames, steps_per_frame, filename):
    with VideoWriter(filename=filename) as vid, torch.no_grad():
        x = ca_model.seed(1, size)

        for k in tnrange(num_frames, leave=False):
            for i in range(steps_per_frame):
                x[:] = ca_model(x)

                #to rgb and move to cpu
                img = x[0][...,:3,:,:].permute(1, 2, 0).cpu()
                img += 0.5
                img = np.uint8(img.clip(0, 1)*255)
            vid.add(img)

    return(filename)
    

def show_video(video_path, video_width = 600):
   
    video_file = open(video_path, "r+b").read()
    video_url = f"data:video/mp4;base64,{b64encode(video_file).decode()}"
    return HTML(f"""<video width={video_width} controls><source src="{video_url}"></video>""")


def C(ca_model, size, num_frames, steps_per_frame, fps, filename):
    gif_arr=np.zeros((600, size,size, 3))

    with torch.no_grad():
        x = ca_model.seed(1, size)

        for k in tnrange(num_frames, leave=False):
            for i in range(steps_per_frame):
                x[:] = ca_model(x)
            img = to_rgb(x[0]).permute(1, 2, 0).cpu()
            gif_arr[k]=img

    with imageio.get_writer(filename, mode='I', fps=fps) as writer:
        for out in gif_arr:
            writer.append_data(out)
        writer.close()

    return(filename)
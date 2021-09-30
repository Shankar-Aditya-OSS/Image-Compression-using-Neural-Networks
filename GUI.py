from __future__ import print_function
import os
import tkinter as tk
from tkinter import filedialog
import encode,decode
import numpy as np
from scipy import signal
from scipy.ndimage.filters import convolve
from PIL import Image
import shutil 


window= tk.Tk()
window.title("Image Compressor")
window.wm_iconbitmap('D:/Major-Project/Mine/Code/functions/icon.ico')
dialog_title = 'QUIT'
dialog_text = 'Are you sure?'


window.geometry('1000x600')
window.configure(background='teal')
window.resizable(False, False)
window.grid_rowconfigure(0, weight=1)
window.grid_rowconfigure(1, weight=1)
window.grid_columnconfigure(0, weight=1)
window.grid_columnconfigure(1, weight=1)


def evaluate(org_img,cmp_img):
    def _FSpecialGauss(size, sigma):
        """Function to mimic the 'fspecial' gaussian MATLAB function."""
        radius = size // 2
        offset = 0.0
        start, stop = -radius, radius + 1
        if size % 2 == 0:
            offset = 0.5
            stop -= 1
        x, y = np.mgrid[offset + start:stop, offset + start:stop]
        assert len(x) == size
        g = np.exp(-((x**2 + y**2) / (2.0 * sigma**2)))
        return g / g.sum()
    def _SSIMForMultiScale(img1,
                           img2,
                           max_val=255,
                           filter_size=11,
                           filter_sigma=1.5,
                           k1=0.01,
                           k2=0.03):
   
        if img1.shape != img2.shape:
            raise RuntimeError(
                'Input images must have the same shape (%s vs. %s).', img1.shape,
                img2.shape)
        if img1.ndim != 4:
            raise RuntimeError('Input images must have four dimensions, not %d',
                               img1.ndim)
            
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        _, height, width, _ = img1.shape
        
        # Filter size can't be larger than height or width of images.
        size = min(filter_size, height, width)
        
        # Scale down sigma if a smaller filter size is used.
        sigma = size * filter_sigma / filter_size if filter_size else 0
        
        if filter_size:
            window = np.reshape(_FSpecialGauss(size, sigma), (1, size, size, 1))
            mu1 = signal.fftconvolve(img1, window, mode='valid')
            mu2 = signal.fftconvolve(img2, window, mode='valid')
            sigma11 = signal.fftconvolve(img1 * img1, window, mode='valid')
            sigma22 = signal.fftconvolve(img2 * img2, window, mode='valid')
            sigma12 = signal.fftconvolve(img1 * img2, window, mode='valid')
        else:
            # Empty blur kernel so no need to convolve.
            mu1, mu2 = img1, img2
            sigma11 = img1 * img1
            sigma22 = img2 * img2
            sigma12 = img1 * img2
            
        mu11 = mu1 * mu1
        mu22 = mu2 * mu2
        mu12 = mu1 * mu2
        sigma11 -= mu11
        sigma22 -= mu22
        sigma12 -= mu12
        
        # Calculate intermediate values used by both ssim and cs_map.
        c1 = (k1 * max_val)**2
        c2 = (k2 * max_val)**2
        v1 = 2.0 * sigma12 + c2
        v2 = sigma11 + sigma22 + c2
        ssim = np.mean((((2.0 * mu12 + c1) * v1) / ((mu11 + mu22 + c1) * v2)))
        cs = np.mean(v1 / v2)
        return ssim, cs
    
    def MultiScaleSSIM(img1,
                       img2,
                       max_val=255,
                       filter_size=11,
                       filter_sigma=1.5,
                       k1=0.01,
                       k2=0.03,
                       weights=None):
     
        if img1.shape != img2.shape:
            raise RuntimeError(
                'Input images must have the same shape (%s vs. %s).', img1.shape,img2.shape)
        if img1.ndim != 4:
            raise RuntimeError('Input images must have four dimensions, not %d',
                               img1.ndim)
            
        # Note: default weights don't sum to 1.0 but do match the paper / matlab
        # code.
        
        weights = np.array(weights if weights else
                       [0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
        levels = weights.size
        downsample_filter = np.ones((1, 2, 2, 1)) / 4.0
        im1, im2 = [x.astype(np.float64) for x in [img1, img2]]
        mssim = np.array([])
        mcs = np.array([])
        
        for _ in range(levels):
            ssim, cs = _SSIMForMultiScale(
                im1,
                im2,
                max_val=max_val,
                filter_size=filter_size,
                filter_sigma=filter_sigma,
                k1=k1,
                k2=k2)
            mssim = np.append(mssim, ssim)
            mcs = np.append(mcs, cs)
            filtered = [
                convolve(im, downsample_filter, mode='reflect')
                for im in [im1, im2]
            ]
            im1, im2 = [x[:, ::2, ::2, :] for x in filtered]
        return (np.prod(mcs[0:levels - 1]**weights[0:levels - 1]) *
                (mssim[levels - 1]**weights[levels - 1]))
    
    def msssim(original, compared):
        if isinstance(original, str):
            original = np.array(Image.open(
                original).convert('RGB'), dtype=np.float32)
        if isinstance(compared, str):
            compared = np.array(Image.open(
                compared).convert('RGB'), dtype=np.float32)
            
        original = original[None, ...] if original.ndim == 3 else original
        compared = compared[None, ...] if compared.ndim == 3 else compared
        
        return MultiScaleSSIM(original, compared, max_val=255)
    
    def main():
        original_image = org_img
        compared_image = cmp_img
        output_msssim = 'SSIM:', msssim(original_image,compared_image)
        message.configure(text=output_msssim)
        source = 'D:/Major-Project/Mine/Code/' + filename[1]
        destination = 'D:\Major-Project\Mine\Code\output'
        dest = 'D:/Major-Project/Mine/Code/output/' +filename[1] 
        if os.path.exists(dest):
            os.remove(dest)
        move =  shutil.move(source, destination)
        source1 = 'D:/Major-Project/Mine/Code/' + filename[2]
        dest1 = 'D:/Major-Project/Mine/Code/output/' + filename[2]
        if os.path.exists(dest1):
            os.remove(dest1)
        move1 =  shutil.move(source1, destination)
        filename.clear()
        fname.clear()
        
    if __name__ == '__main__':
        main()
        


fname=[]
def browseFiles():
    message.configure(text='')
    message1.configure(text='')
    message2.configure(text='---')
    message3.configure(text='---')
    message4.configure(text='---')
    filename = filedialog.askopenfilename(initialdir = "/D:/Major-Project/Mine/Code",
                                          title = "Select a File",
                                          filetypes = (("Images",
                                                        "*.png*"),
                                                       ("all files",
                                                       "*.*")))
    name=str(filename[-4:])
    if name!='.png':
        filename=None
    else:
        message1.configure(text=filename)
        fname.append(filename)
        input_filesize = os.path.getsize(filename)
        val = round(input_filesize/1024)
        fname.append(str(val))
        message2.configure(text=fname[1]+' KB')
    



filename=[]
def encoder():
    try:
        inp = fname[0]
        img_= inp[35:-4]
        img = str(img_)
        print(img)
        filename.append(img)
        encode.encode(inp,img)
        message.configure(text='Created the Binary file')
    except:
        message.configure(text='Please choose an input image...')
    filename1 = str(filename[0])
    filename2 = filename1 + '(compressed).npz'
    filename.append(filename2)
    output_filesize2 = os.path.getsize(filename2)
    val2 = round(output_filesize2/1024)
    fname.append(str(val2))
    message3.configure(text=fname[2]+' KB') 


def decoder():
    try:
        filename1 = str(filename[0])
    except:
        message.configure(text='Please enocde the image first...')
    
    print(filename)
    inp1 = filename1 + '(compressed).npz'
    decode.decode(inp1,filename1)
    message.configure(text='Image Reconstructed')
    filename3= filename1 + '(compressed).png'
    output_filesize = os.path.getsize(filename3)
    val1 = round(output_filesize/1024)
    fname.append(str(val1))
    message4.configure(text=fname[3]+' KB')
    
 
def msssim():
    try:
        filename1 = str(filename[0])
        org_img = fname[0]
        cmp_img = filename1 + '(compressed).png'
        filename.append(cmp_img)
        evaluate(org_img,cmp_img)
    except:
        message.configure(text='Please provide an input/output image...')
    


message1 = tk.Label(window, text='' ,bg="yellow"  ,fg="black"  ,width=50 ,height=2, activebackground = "yellow" ,font=('times', 15, ' bold ')) 
message1.place(x=285, y=185)

message = tk.Label(window, text="Image Compression using Neural Networks" ,bg="lime"  ,fg="black"  ,width=50  ,height=3,font=('times', 25, 'italic bold underline')) 
message.place(x=23, y=20)

lbl1 = tk.Label(window, text="Original Image : ",width=17  ,fg="black"  ,bg="yellow"  ,height=2 ,font=('times', 13, ' bold underline ')) 
lbl1.place(x=285, y=265)

lbl2 = tk.Label(window, text="Binary File : ",width=17  ,fg="black"  ,bg="yellow"  ,height=2 ,font=('times', 13, ' bold underline ')) 
lbl2.place(x=285, y=300)

lbl3 = tk.Label(window, text="Reconstructed Image : ",width=17  ,fg="black"  ,bg="yellow"  ,height=2 ,font=('times', 13, ' bold underline ')) 
lbl3.place(x=285, y=333)

lbl4 = tk.Label(window, text="Notification : ",width=15  ,fg="black"  ,bg="yellow"  ,height=2 ,font=('times', 15, ' bold underline ')) 
lbl4.place(x=70, y=400)

message = tk.Label(window, text="" ,bg="yellow"  ,fg="black"  ,width=35 ,height=2, activebackground = "yellow" ,font=('times', 15, ' bold ')) 
message.place(x=285, y=400)

message2 = tk.Label(window, text="---" ,bg="yellow"  ,fg="black"  ,width=12 ,height=2, activebackground = "yellow" ,font=('times', 13, ' bold ')) 
message2.place(x=460, y=265)
message3 = tk.Label(window, text="---" ,bg="yellow"  ,fg="black"  ,width=12 ,height=2, activebackground = "yellow" ,font=('times', 13, ' bold ')) 
message3.place(x=460, y=301)
message4 = tk.Label(window, text="---" ,bg="yellow"  ,fg="black"  ,width=12 ,height=2, activebackground = "yellow" ,font=('times', 13, ' bold ')) 
message4.place(x=460, y=333)

browseButton = tk.Button(window, text="Browse", command=browseFiles ,width=15  ,height=2  ,fg="black"  ,bg="yellow" ,font=('times', 15, ' bold '),borderwidth=4, relief="solid") 
browseButton.place(x=70, y=180)  

Encode = tk.Button(window, text="Encode", command=encoder  ,fg="black"  ,bg="yellow"  ,width=20 ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '),borderwidth=4, relief="solid")
Encode.place(x=70, y=480)
Decode = tk.Button(window, text="Decode", command=decoder  ,fg="black"  ,bg="yellow"  ,width=20 ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '),borderwidth=4, relief="solid")
Decode.place(x=390, y=480)
Eval = tk.Button(window, text="Eval", command=msssim  ,fg="black"  ,bg="yellow"  ,width=20 ,height=2, activebackground = "Red" , font=('times', 15, ' bold '),borderwidth=4, relief="solid")
Eval.place(x=710, y=480)
copyWrite = tk.Text(window, background=window.cget("background"), borderwidth=0,font=('times', 30, 'italic bold underline'),)
copyWrite.configure(state="disabled",fg="red"  )
copyWrite.place(x=800, y=750)



window.mainloop()

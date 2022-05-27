import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras import backend as K
#from keras.engine.topology import Layer
from tensorflow.keras.layers import Layer
import numpy as np
import cv2
import math
import scipy.ndimage

from tensorflow.python.keras.utils.data_utils import Sequence
      
tf.keras.backend.set_floatx('float32')


class Diffract(tf.keras.layers.Layer):

    def __init__(self, output_dim, wl=633e-9, p=10e-6, z=-0.1, **kwargs):
        self.wl = wl
        self.p = p
        self.z = z
        self.output_dim = output_dim

        super(Diffract, self).__init__(**kwargs)

    def build(self, input_shape):
        self.shape = input_shape
        #self.batch = input_shape[0]
        self.nx = input_shape[2]
        self.ny = input_shape[1]
        self.ch = input_shape[3]
        self.nx2 = self.nx * 2
        self.ny2 = self.ny * 2
        self.px =  1 / (self.nx2 * self.p)
        self.py =  1 / (self.ny2 * self.p)
        super(Diffract, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        #f=tf.squeeze(x, axis=-1)

        f=tf.complex(x[:,:,:,0],x[:,:,:,1])
        
        ### zero padding
        #f=Lambda(lambda v:  tf.pad(v,[[0,0],[self.ny//2,self.ny//2],[self.nx//2,self.nx//2]]))(f)
        f=tf.pad(f,[[0,0],[self.ny//2,self.ny//2],[self.nx//2,self.nx//2]])

        # print("xxx before:",f.shape)
        #f = Lambda(lambda v: tf.signal.fft2d(tf.cast(v, tf.complex64)))(f)
        f=tf.signal.fft2d(tf.cast(f, tf.complex64))
        
        x, y = tf.meshgrid(tf.linspace(-self.ny2/2, self.ny2/2, tf.cast(self.ny2, tf.int32)),
                           tf.linspace(-self.nx2/2, self.nx2/2, tf.cast(self.nx2, tf.int32)))
        fx = tf.cast(x,tf.float64)*self.px
        fy = tf.cast(y,tf.float64)*self.py
        ph = tf.exp(tf.dtypes.complex(tf.dtypes.cast(0.,tf.float64),+2*np.pi*self.z*tf.sqrt(1/(self.wl*self.wl)-(fx**2+fy**2))))
        ph = tf.cast(ph, tf.complex64)
        ph=tf.signal.fftshift(ph)
        #print(ph.shape)
        #print(f.shape)
        #print(ph)

        #f = Lambda(lambda v: tf.math.multiply(v[0], v[1]))([f,ph]) 
        f = tf.math.multiply(f, ph)
        
        #f = Lambda(lambda v: tf.signal.ifft2d(tf.cast(v, tf.complex64)))(f)
        f = tf.signal.ifft2d(tf.cast(f, tf.complex64))

        #print("!!diffract_layer shape : ", f)
        #f = Lambda(lambda v: tf.slice(v,(0,self.ny//2,self.nx//2),(-1,self.ny,self.nx)))(f)
        f = tf.slice(f, (0, self.ny//2, self.nx//2), (-1, self.ny, self.nx))

        #f = Lambda(lambda v: tf.concat([tf.expand_dims(tf.math.real(v), axis=-1), tf.expand_dims(tf.math.imag(v), axis=-1)], 3))(f)
        f = tf.concat([tf.expand_dims(tf.math.real(f), axis=-1), tf.expand_dims(tf.math.imag(f), axis=-1)], 3)
        #f = Lambda(lambda v: tf.abs(v))(f)
        #f = Lambda(lambda v: tf.pow(v, 2.0))(f)
        #f = Lambda(lambda v: tf.cast(v,tf.float32))(f)

        #f=tf.expand_dims(f,axis=-1)
      
        return f

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


class ComplexBias(tf.keras.layers.Layer):
    def __init__(self, kernel_shape):     
        #print('ComplexBias __init__')
        #print('self.kernel_shape : ', kernel_shape)
        super(ComplexBias, self).__init__(dtype='float32')
        self.kernel_shape = kernel_shape
        #print('ComplexBias __init__ end')
        
    def build(self, input_shape):
        #print('ComplexBias build')
        #print('self.kernel_shape : ', self.kernel_shape)
        #print('input_shape : ', input_shape)
        super(ComplexBias, self).build(input_shape)
        self.kernel = self.add_weight(name="kernel", 
                                      dtype=tf.float32,
                                      #constraint=tf.keras.constraints.MinMaxNorm(min_value=0.0, max_value=2*np.pi, rate=1.0, axis=0),
                                      #constraint=tf.keras.constraints.NonNeg(),
                                      shape=(self.kernel_shape[0], self.kernel_shape[1]),
                                      #initializer=tf.keras.initializers.Constant(np.pi)
                                      initializer=tf.keras.initializers.RandomUniform(minval=0, maxval=2.0*np.pi)
                                     )
        #print('self.kernel.shape : ', self.kernel.shape)
        #print('ComplexBias build end')

        
    def call(self, input):
        #print('ComplexBias call')
        
        real = tf.keras.backend.cos(self.kernel)
        imag = tf.keras.backend.sin(self.kernel)
        #print('real.dtype : ', real.dtype)
        #print('input : ', input)
            
        out_real = tf.multiply(input[:, :, :, 0], real) - tf.multiply(input[:, :, :, 1], imag)
        out_imag = tf.multiply(input[:, :, :, 0], imag) + tf.multiply(input[:, :, :, 1], real)
        out_real = tf.reshape(out_real, [-1, out_real.shape[1], out_real.shape[2], 1])
        out_imag = tf.reshape(out_imag, [-1, out_imag.shape[1], out_imag.shape[2], 1])

        #print('real : ', real)
        #print('out_real.shape : ', out_real.shape)
        #print('out_imag.shape : ', out_imag.shape)
        
        out = tf.concat([out_real, out_imag], 3)
        
        out_max = tf.math.reduce_max(out)
        out = out / out_max
        
        #print('out.shape : ', out.shape)
        #print('out.dtype : ', out.dtype)
        #print('out_real.dtype : ', out_real.dtype)
        #print(tf.reshape(out_real, [neuron_height_num, neuron_width_num]).shape)
   
        
        #print('ComplexBias call end')
        return out
    
class Intensity(tf.keras.layers.Layer):
    def __init__(self):
        super(Intensity, self).__init__()
        
    def bulid(self, input_shape):
        super(Intensity, self).build(input_shape)
        
    def call(self, input):
        #print('input shape : ', input.shape)
        intensity = tf.multiply(input, input)
        intensity = tf.math.reduce_sum(intensity, axis=3)
        intens_max = tf.math.reduce_max(intensity)
        intensity = intensity / intens_max * 255
        #intensity = tf.reshape(input[0], input[1], input[2])
        #print('final shape : ', intensity.shape)
        return intensity
    
def custom_loss_by_label(detect_mask, height, width): 
    #teacher_max = 255
    #teacher_mask = y_true / teacher_max
    #y_pred = tf.multiply(y_pred, teacher_mask)
    
    def loss_function(y_true, y_pred):
        y_pred = tf.multiply(y_pred, detect_mask)
    
        tmp = y_pred - y_true
        tmp = tf.multiply(tmp, tmp)
        tmp = tf.reduce_sum(tmp, axis=1)
        tmp = tf.reduce_sum(tmp, axis=1)
        error = tmp / (height * width)
        return error
    return loss_function
    
def custom_accuracy(teacher_mask):
    import tensorflow.keras.backend as K
    
    def accuracy(y_true, y_pred):
        height = len(teacher_mask[0])
        width = len(teacher_mask[0][0])
        
        true_tmp = tf.multiply(y_true[:], teacher_mask[0])
        true_tmp = tf.reshape(true_tmp, [-1, height, width, 1])
        true_tmp = tf.reduce_sum(true_tmp, axis=1)
        true_tmp = tf.reduce_sum(true_tmp, axis=1)
        
        pred_tmp = tf.multiply(y_pred[:], teacher_mask[0])
        pred_tmp = tf.reshape(pred_tmp, [-1, height, width, 1])
        pred_tmp = tf.reduce_sum(pred_tmp, axis=1)
        pred_tmp = tf.reduce_sum(pred_tmp, axis=1)
        
        for i in range(1,len(teacher_mask)):
            true_tmp2 = tf.multiply(y_true[:], teacher_mask[i])
            true_tmp2 = tf.reshape(true_tmp2, [-1, height, width, 1])
            true_tmp2 = tf.reduce_sum(true_tmp2, axis=1)
            true_tmp2 = tf.reduce_sum(true_tmp2, axis=1)
            true_tmp = tf.concat([true_tmp, true_tmp2], 1)

            pred_tmp2 = tf.multiply(y_pred[:], teacher_mask[i])
            pred_tmp2 = tf.reshape(pred_tmp2, [-1, height, width, 1])
            pred_tmp2 = tf.reduce_sum(pred_tmp2, axis=1)
            pred_tmp2 = tf.reduce_sum(pred_tmp2, axis=1)
            pred_tmp = tf.concat([pred_tmp, pred_tmp2], 1)

        true_label = tf.math.argmax(true_tmp,1)
        pred_label = tf.math.argmax(pred_tmp,1)

        return K.cast(K.equal(true_label, pred_label), K.floatx())
    return accuracy

class DataSequence(Sequence):
    def __init__(self, data_path, teacher_path, data_range, data_pattern_num, detector_order, data_expand, batch_size=32, img_random_offset=None, img_offset=None):
        self.data_path = data_path
        self.teacher_path = teacher_path
        self.data_range = data_range
        self.data_size = len(self.data_range)
        self.data_pattern_num = data_pattern_num
        self.detector_order = detector_order
        self.data_expand = data_expand
        self.batch_size = batch_size
        self.length = self.data_size // self.batch_size
        self.img_random_offset = img_random_offset
        self.img_offset = img_offset
    
    def __len__(self):
 
        return self.length
    
    def __getitem__(self, idx):
        batch_lead = idx * self.batch_size
        X = []
        Y = []
        for i in range(batch_lead, batch_lead+self.batch_size, 1):
            tmp = np.load(self.data_path.format(self.data_range[i]))
            shift_flag = False
            x_shift = 0
            y_shift = 0
            if self.img_random_offset:
                np.random.seed(seed=i)
                rnd_tmp = np.random.randint(-self.img_random_offset, self.img_random_offset+1, 2)
                x_shift = rnd_tmp[0]
                y_shift = rnd_tmp[1]
                shift_flag = True
            elif self.img_offset:
                x_shift = self.img_offset[0]
                y_shift = self.img_offset[1]
                shift_flag = True
            if shift_flag:                
                tmp = np.roll(tmp, x_shift, axis=1)
                tmp = np.roll(tmp, y_shift, axis=0)
                if x_shift >= 0:
                    tmp[:, :x_shift] = 0 + 0j
                else:
                    tmp[:, x_shift:] = 0 + 0j
                if y_shift >= 0:
                    tmp[:y_shift, :] = 0 + 0j
                else:
                    tmp[y_shift:, :] = 0 + 0j
           
                
            X.append(scipy.ndimage.zoom(tmp, self.data_expand, order=1))
            
            y = self.detector_order[i % self.data_pattern_num]
            Y.append(cv2.imread(self.teacher_path.format(y), cv2.IMREAD_GRAYSCALE))
        
        X = np.asarray(X)
        X_real = np.real(X)
        X_imag = np.imag(X)
        X_real = X_real[:,:,:,np.newaxis]
        X_imag = X_imag[:,:,:,np.newaxis]
        X = np.append(X_real, X_imag, axis=3)#.repeat(self.data_expand, axis=1).repeat(self.data_expand, axis=2)
        Y = np.asarray(Y, dtype='float32')#.repeat(self.data_expand, axis=1).repeat(self.data_expand, axis=2)

        return X, Y
    
    def on_epoch_end(self):

        pass
    
def create_masks(height, width, detector_width, detector_num, detector_gap, loss_full_surface=False):
    save_dir = './teachers/{}_{}_{}_{}_{}'.format(width, height, detector_width, detector_num, detector_gap)
    
    rc = math.ceil(np.sqrt(detector_num))    
    start_x = 0
    start_y = 0
    gap_x = 0
    gap_y = 0
    
    if detector_gap == None:
        gap_x = (width - detector_width * rc) // (rc + 1)
        gap_y = (height - detector_width * rc) // (rc + 1)
        start_x = gap_x
        start_y = gap_y
    else:
        gap_x = detector_gap
        gap_y = detector_gap
        
        dg = (rc - 1) / 2
        dx = dg + 0.5
        dy = dx
        
        start_x = int(width / 2 - dg * gap_x - dx * detector_width)
        start_y = int(height / 2 - dg * gap_y - dy * detector_width)
    
    teacher_mask = np.zeros([detector_num, height, width])
    detect_mask = np.zeros([height, width])
    for i in range(detector_num):
        dx = i % rc
        dy = i // rc
        
        teacher = np.zeros([height, width])
        
        teacher[start_y+(detector_width+gap_y)*dy:start_y+detector_width*(dy+1)+gap_y*dy, start_x+(detector_width+gap_x)*dx:start_x+detector_width*(dx+1)+gap_x*dx] = 255
        detect_mask += teacher
        teacher_mask[i] = teacher
        cv2.imwrite(save_dir + '/{:0>2}.png'.format(i), teacher)
    if loss_full_surface == False:
        detect_mask = np.where(detect_mask == 0, 0, 1)
    else:
        detect_mask = np.where(detect_mask == 0, 1, 1)
    
    return detect_mask, teacher_mask

class ComplexBias_AnyKernel(tf.keras.layers.Layer):
    def __init__(self, kernel_shape, weights):     
        #print('ComplexBias __init__')
        #print('self.kernel_shape : ', kernel_shape)
        super(ComplexBias_AnyKernel, self).__init__(dtype='float32')
        self.kernel_shape = kernel_shape
        self.kernel = weights
        #print('ComplexBias __init__ end')
        
    def build(self, input_shape):
        #print('ComplexBias_AnyKernel build')
        #print('self.kernel_shape : ', self.kernel_shape)
        #print('input_shape : ', input_shape)
        super(ComplexBias_AnyKernel, self).build(input_shape)
        #self.kernel = self.add_weight(name="kernel", 
        #                              dtype=tf.float32,
        #                              #constraint=tf.keras.constraints.MinMaxNorm(min_value=0.0, max_value=2*np.pi, rate=1.0, axis=0),
        #                              #constraint=tf.keras.constraints.NonNeg(),
        #                              shape=(self.kernel_shape[0], self.kernel_shape[1]),
        #                              #initializer=tf.keras.initializers.Constant(np.pi)
        #                              initializer=tf.keras.initializers.RandomUniform(minval=0, maxval=2.0*np.pi)
        #                             )
        #print('self.kernel.shape : ', self.kernel.shape)
        #print('ComplexBias build end')

        
    def call(self, input):
        #print('ComplexBias_AnyKernel call')
        
        real = tf.keras.backend.cos(self.kernel)
        imag = tf.keras.backend.sin(self.kernel)
        #print('real.dtype : ', real.dtype)
        #print('input : ', input)
            
        out_real = tf.multiply(input[:, :, :, 0], real) - tf.multiply(input[:, :, :, 1], imag)
        out_imag = tf.multiply(input[:, :, :, 0], imag) + tf.multiply(input[:, :, :, 1], real)
        out_real = tf.reshape(out_real, [-1, out_real.shape[1], out_real.shape[2], 1])
        out_imag = tf.reshape(out_imag, [-1, out_imag.shape[1], out_imag.shape[2], 1])

        #print('real : ', real)
        #print('out_real.shape : ', out_real.shape)
        #print('out_imag.shape : ', out_imag.shape)
        
        out = tf.concat([out_real, out_imag], 3)
        
        out_max = tf.math.reduce_max(out)
        out = out / out_max
        
        #print('out.shape : ', out.shape)
        #print('out.dtype : ', out.dtype)
        #print('out_real.dtype : ', out_real.dtype)
        #print(tf.reshape(out_real, [neuron_height_num, neuron_width_num]).shape)
   
        
        #print('ComplexBias call end')
        return out
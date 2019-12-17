import tensorflow as tf
import cv2 as cv
import numpy as np
import math
from MotionTrackingLK import gaussian
import MotionTrackingLK as mtlk


class SparseOptFlowLK(tf.keras.layers.Layer):
    def __init__(self, window_pixel_wh=21, sigma=2, iterations=5, **kwargs):
        self.sigma = sigma
        assert(window_pixel_wh >= 3)
        self.win_pixel_wh = window_pixel_wh
        self.iterations = iterations
        super(SparseOptFlowLK, self).__init__(**kwargs)

    def build(self, input_shape):
        # grab the dimensions of the image here so we can use them later. also will throw errors early for users
        self.h = input_shape[2]
        self.w = input_shape[3]
        self.c = input_shape[4]

        v_splits = self.h//self.win_pixel_wh
        h_splits = self.w//self.win_pixel_wh

        self.num_tracks = v_splits*h_splits

        win_xs, win_ys = tf.meshgrid(
            tf.range(h_splits, dtype=tf.float32)*self.win_pixel_wh,
            tf.range(v_splits, dtype=tf.float32)*self.win_pixel_wh
        )
        win_xs, win_ys = mtlk.device_to_logical(self.w, self.h, win_xs+self.win_pixel_wh//2, win_ys+self.win_pixel_wh//2)
        win_positions = tf.stack([
            win_xs,
            win_ys
        ], axis=-1)
        self.win_positions = tf.reshape(win_positions, [1, -1, 1, 2])

        self.center_relative = tf.constant(
            [self.w/self.win_pixel_wh, self.h/self.win_pixel_wh],
            shape=[1,1,2,1]
        )

        self.sobel_x = tf.constant([
                [-1.,  0.,  1.],
                [-2.,  0.,  2.],
                [-1.,  0.,  1.],
            ],
            shape=[3, 3, 1, 1]
        )
        self.sobel_y = tf.constant([
                [-1., -2., -1.],
                [ 0.,  0.,  0.],
                [ 1.,  2.,  1.],
            ],
            shape=[3, 3, 1, 1]
        )

        self.scharr_x = tf.constant([
                [-3.,   0.,  3.],
                [-10.,  0.,  10.],
                [-3.,   0.,  3.],
            ],
            shape=[3, 3, 1, 1]
        )
        self.scharr_y = tf.constant([
                [-3., -10., -3.],
                [ 0.,   0.,  0.],
                [ 3.,  10.,  3.],
            ],
            shape=[3, 3, 1, 1]
        )

        weights = np.empty([self.win_pixel_wh, self.win_pixel_wh])
        center = self.win_pixel_wh//2
        for y in range(self.win_pixel_wh):
            for x in range(self.win_pixel_wh):
                weights[y, x] = (x-center)**2 + (y-center)**2

        weights = gaussian(np.sqrt(weights), self.sigma)
        self.win_weights = tf.constant(weights, shape=[1, 1, self.win_pixel_wh*self.win_pixel_wh, 1], dtype=tf.float32)
        # print(weights)
        # tf.print(weights)
        # tf.print(tf.reduce_max(weights))

        super(SparseOptFlowLK, self).build(input_shape)

    def calc_velocity_2frames_ntracks_LK(self, first_frame, second_frame, ff_dx, ff_dy):
        ff_comb = tf.reshape(first_frame, [-1, self.win_pixel_wh, self.win_pixel_wh, self.c])

        Ix = tf.reshape(
            ff_dx,
            [-1, self.num_tracks, self.win_pixel_wh*self.win_pixel_wh, 1]
        )

        Iy = tf.reshape(
            ff_dy,
            [-1, self.num_tracks, self.win_pixel_wh*self.win_pixel_wh, 1]
        )

        A = tf.concat([Ix,Iy], axis=3)
        ATA = tf.matmul(A, A*self.win_weights, transpose_a=True)

        # ATA_1 = tf.linalg.inv(ATA)
        # tf.linalg.inv gives me a cusolver error, so i generate inverse manually
        a = ATA[:,:, 0,0]
        b = ATA[:,:, 0,1]
        c = ATA[:,:, 1,0]
        d = ATA[:,:, 1,1]
        ATA_1 = tf.reshape(
            tf.math.divide_no_nan(1.0, a*d - b*c),
            [-1, self.num_tracks, 1, 1])*tf.stack([tf.stack([d, -b], axis=-1), tf.stack([-c, a], axis=-1)], axis=3
        )

        b = -1*tf.reshape(
            second_frame-first_frame,
            [-1, self.num_tracks, self.win_pixel_wh*self.win_pixel_wh, 1]
        )*self.win_weights
        ATb = tf.matmul(A, b, transpose_a=True)

        VxVy = tf.matmul(ATA_1, ATb)

        return VxVy

    def iterative_LK(self, sampler, frames, iterations):
        out = self.sample_ntracks_from_2frames(sampler, frames)
        first_frame = out[:, 0]
        factor = 1.0

        VxVy = self.calc_velocity_2frames_ntracks_LK(first_frame, out[:, 1])*factor
        sampler += VxVy
        sum_VxVy = VxVy

        i = tf.constant(1)
        cond = lambda i, s, f, sf, svv: tf.less(i, iterations)

        def iterate(i, sampler, frames, first_frame, sum_VxVy):
            out = self.sample_ntracks_from_2frames(sampler, frames)

            VxVy = self.calc_velocity_2frames_ntracks_LK(first_frame, out[:, 1])*factor

            sampler += VxVy 
            i += 1
            sum_VxVy += VxVy
            return i, sampler, frames, first_frame, sum_VxVy

        _, sampler, _, _, sum_VxVy = tf.while_loop(cond, iterate, [i, sampler, frames, first_frame, sum_VxVy])
        return sampler, tf.reshape(sum_VxVy, [-1, self.num_tracks, 2])

    def split_img_into_win_grid(self, img):
        n = self.win_pixel_wh
        h_splits = self.h//n
        w_splits = self.w//n

        out = img[:, 0:h_splits*n, 0:w_splits*n]

        out = tf.reshape(
            out,
            [-1, h_splits, n, w_splits, n, self.c]
        )
        out = tf.reshape(
            tf.transpose(out, [0, 1, 3, 2, 4, 5]),
            [-1, h_splits*w_splits, n, n, self.c]
        )

        return out

    def call(self, imgs):
        first_frame = imgs[:, 0]

        ff_dx = tf.nn.convolution(first_frame, self.scharr_x, padding="SAME")
        ff_dy = tf.nn.convolution(first_frame, self.scharr_y, padding="SAME")
        second_frame = imgs[:, 1]

        first_frame = self.split_img_into_win_grid(first_frame)
        ff_dx = self.split_img_into_win_grid(ff_dx)
        ff_dy = self.split_img_into_win_grid(ff_dy)
        second_frame = self.split_img_into_win_grid(second_frame)

        vels = tf.transpose(self.calc_velocity_2frames_ntracks_LK(first_frame, second_frame, ff_dx, ff_dy), [0,1,3,2])
        cr = tf.reshape(self.center_relative, [1,1,1,2])
        vels = vels/cr

        vels = vels + self.win_positions
        return tf.concat([self.win_positions, vels], axis=2)
  
    def compute_output_shape(self, input_shape):
        self.seq_len = input_shape[1][1]
        return [None, self.num_tracks, self.seq_len, 2]
  
    def get_config(self):
        base_config = super(SparseOptFlowLK, self).get_config()
        return base_config
  
    @classmethod
    def from_config(cls, config):
        return cls(**config)

if __name__ == "__main__":
    import numpy as np
    import math

    window_pixel_wh = 15
    sigma = 2
    batches = 1
    iterations = 5

    imgs = np.asarray([
        [
            np.expand_dims(cv.imread("car_dashcam0.png", cv.IMREAD_GRAYSCALE) / 255, axis=-1),
            np.expand_dims(cv.imread("car_dashcam1.png", cv.IMREAD_GRAYSCALE) / 255, axis=-1),
            # np.expand_dims(cv.imread("car_dashcam2.png", cv.IMREAD_GRAYSCALE) / 255, axis=-1),
            # np.expand_dims(cv.imread("car_dashcam3.png", cv.IMREAD_GRAYSCALE) / 255, axis=-1),
            # np.expand_dims(cv.imread("car_dashcam4.png", cv.IMREAD_GRAYSCALE) / 255, axis=-1),
        ]
    ]*batches).astype(np.float32)
    _, _, h, w, c = imgs.shape
    print(w, h ,c)

    out = np.float32(SparseOptFlowLK(
        window_pixel_wh=window_pixel_wh, sigma=sigma, iterations=iterations)(imgs)
    )
    print(out.shape)

    imgs_with_flow = mtlk.display_tracks(imgs, out)

    for img in imgs_with_flow:
        cv.imshow("flow", img)
        cv.waitKey()

    # for batch in out:
    #     for img in batch:
    #         cv.imshow("sdf", img)
    #         cv.waitKey()

    with open("OUT", "w") as f:
        with np.printoptions(threshold=np.inf):
            print("writing to OUT")
            f.write(str(out))
# https://github.com/simon-larsson/keras-swa/blob/master/swa/keras.py
import tensorflow as tf


class SWA:
    def __init__(self, model, start_epoch, swa_freq=2, verbose=True):
        self.model = model
        self.start_epoch = start_epoch - 1
        self.swa_freq = swa_freq
        self.swa_weights = None
        self.cnt = 0
        self.verbose = verbose

    def on_epoch_end(self, epoch):
        epoch = epoch - self.start_epoch
        if epoch == 0 or (epoch > 0 and epoch % self.swa_freq == 0):
            if self.verbose:
                print("\nSaving Weights... ", epoch+self.start_epoch)
            self.update_swa_weights()

    def on_train_end(self):
        print("\nThe final model Has Been set...")
        self.model.set_weights(self.swa_weights)

    def update_swa_weights(self):
        if self.swa_weights is None:
            self.swa_weights = self.model.get_weights()
        else:
            self.swa_weights = [
                (swa_w*self.cnt + w) / (self.cnt+1)
                for swa_w, w in zip(self.swa_weights, self.model.get_weights())]
        self.cnt += 1


import lightning as L


class CustomCallback(L.Callback):
    def on_epoch_end(self, epoch, logs=None):
        pass

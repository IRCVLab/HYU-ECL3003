import gi

gi.require_version("Gst", "1.0")
from gi.repository import GObject, Gst

import time
import numpy as np
from threading import Thread


def _sanitize(element) -> Gst.Element:
    """
    Passthrough function which sure element is not `None`
    Returns `Gst.Element` or raises Error
    """
    if element is None:
        raise Exception("Element is none!")
    else:
        return element


def _make_element_safe(el_type: str, el_name=None) -> Gst.Element:
    """
    Creates a gstremer element using el_type factory.
    Returns Gst.Element or throws an error if we fail.
    This is to avoid `None` elements in our pipeline
    """

    # name=None parameter asks Gstreamer to uniquely name the elements for us
    el = Gst.ElementFactory.make(el_type, name=el_name)

    if el is not None:
        return el
    else:
        print(f"Pipeline element is None!")
        raise NameError(f"Could not create element {el_type}")


class Camera:
    def __init__(self, sensor_id, fps=30, shape_in=(1920, 1080), shape_out=(1920, 1080)) -> None:

        GObject.threads_init()
        Gst.init(None)
        self._mainloop = GObject.MainLoop()
        self._pipeline = self._make_pipeline_with_resize(
            sensor_id, fps, shape_in, shape_out, format="GRAY8"
        )
        self._pipeline.set_state(Gst.State.PLAYING)
        self.wait_ready()

    def stop(self):
        self._pipeline.set_state(Gst.State.NULL)

    def _make_pipeline_with_resize(
        self, sensor_id, fps=None, shape_in=None, shape_out=None, format="GRAY8"
    ):

        pipeline = _sanitize(Gst.Pipeline())

        # Camera
        camera = _make_element_safe("nvarguscamerasrc")
        camera.set_property("sensor-id", sensor_id)

        # Input CF (NVMM / NV12)
        camera_cf = self._make_input_capsfilter(fps, shape_in)

        # nvvidconv: hardware NV12(NVMM) -> BGRx(system memory).
        # Direct GRAY8 from nvvidconv is unreliable on L4T R35 -> use BGRx then videoconvert.
        nvvidconv = _make_element_safe("nvvidconv")
        nvvidconv_cf = self._make_nvvidconv_capsfilter(shape_out)

        # videoconvert: BGRx -> final format (e.g. GRAY8) on CPU.
        vidconv = _make_element_safe("videoconvert")
        appsink_cf = self._make_output_capsfilter(shape_out, format)

        # Appsink
        self._appsink = appsink = _make_element_safe("appsink")
        appsink.set_property("drop", True)
        appsink.set_property("max-buffers", 1)

        for el in [camera, camera_cf, nvvidconv, nvvidconv_cf, vidconv, appsink_cf, appsink]:
            pipeline.add(el)

        camera.link(camera_cf)
        camera_cf.link(nvvidconv)
        nvvidconv.link(nvvidconv_cf)
        nvvidconv_cf.link(vidconv)
        vidconv.link(appsink_cf)
        appsink_cf.link(appsink)

        return pipeline

    @staticmethod
    def _make_nvvidconv_capsfilter(shape_out):
        caps_str = "video/x-raw, format=(string)BGRx"
        if shape_out:
            W_out, H_out = shape_out
            caps_str += f", width={W_out}, height={H_out}"
        caps = Gst.Caps.from_string(caps_str)
        cf = _make_element_safe("capsfilter")
        cf.set_property("caps", caps)
        return cf

    def _make_pipeline(self, sensor_id):

        pipeline = _sanitize(Gst.Pipeline())

        cam = _make_element_safe("nvarguscamerasrc")
        cam.set_property("sensor-id", sensor_id)

        conv = _make_element_safe("nvvidconv")

        cf = _make_element_safe("capsfilter")
        cf.set_property(
            "caps", Gst.Caps.from_string("video/x-raw, format=(string)GRAY8")
        )

        self._appsink = appsink = _make_element_safe("appsink")

        for el in [cam, conv, cf, appsink]:
            pipeline.add(el)

        cam.link(conv)
        conv.link(cf)
        cf.link(appsink)

        return pipeline

    @staticmethod
    def _make_input_capsfilter(fps, shape_in):

        caps_str = "video/x-raw(memory:NVMM), format=(string)NV12"

        if shape_in:
            W_in, H_in = shape_in
            caps_str += f", width=(int){W_in}, height=(int){H_in}"
        if fps:
            caps_str += f", framerate=(fraction){fps}/1"

        caps = Gst.Caps.from_string(caps_str)
        in_cf = _make_element_safe("capsfilter")
        in_cf.set_property("caps", caps)

        return in_cf

    @staticmethod
    def _make_output_capsfilter(shape_out, format):
        caps_str = "video/x-raw"

        if shape_out:
            W_out, H_out = shape_out
            caps_str += f", width={W_out}, height={H_out}"

        if format == "BGR":
            caps_str += ", format=(string)BGRx"
        elif format == "RGBA":
            caps_str += ", format=(string)RGBA"
        elif format in ("GRAY", "GRAY8"):
            caps_str += ", format=(string)GRAY8"

        caps = Gst.Caps.from_string(caps_str)
        cf = _make_element_safe("capsfilter")
        cf.set_property("caps", caps)
        return cf

    def read(self):
        """
        Returns np.array (H, W) for GRAY8 or (H, W, 3) for BGRx/RGBA, or None.
        """
        sample = self._appsink.emit("pull-sample")
        if sample is None:
            return None
        buf = sample.get_buffer()
        caps_struct = sample.get_caps().get_structure(0)
        W = caps_struct.get_value("width")
        H = caps_struct.get_value("height")
        fmt = caps_struct.get_value("format")
        buf2 = buf.extract_dup(0, buf.get_size())

        if fmt == "GRAY8":
            return np.ndarray(shape=(H, W), buffer=buf2, dtype=np.uint8)
        arr = np.ndarray(shape=(H, W, 4), buffer=buf2, dtype=np.uint8)
        return arr[:, :, :3]

    def running(self):
        _, state, _ = self._pipeline.get_state(1)
        return True if state == Gst.State.PLAYING else False

    def wait_ready(self):
        while not self.running():
            time.sleep(0.1)

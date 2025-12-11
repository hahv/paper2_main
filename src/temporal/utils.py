import seaborn as sns
import importlib


def getColor(classIdx):
    # main palette
    palette = sns.color_palette(palette="gist_rainbow")
    color = palette[classIdx % len(palette)]
    # Convert to 255 scale
    r, g, b = color
    color_255 = (int(r * 255), int(g * 255), int(b * 255))
    return color_255


def bgr_to_rgb(bgr):
    """Convert BGR color to RGB."""
    return (bgr[2], bgr[1], bgr[0])


def validate_detect_frame_return(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)

        # Unpack return
        if not isinstance(result, tuple) or len(result) != 3:
            raise ValueError(
                "detect_frame must return a 3-tuple: (shouldDoInfer, infer_result_dict, vis_data_results)"
            )

        shouldDoInfer, infer_result_dict, vis_data_results = result

        if not isinstance(shouldDoInfer, bool):
            raise TypeError("shouldDoInfer must be a bool")

        if infer_result_dict is not None and not isinstance(infer_result_dict, dict):
            raise TypeError("infer_result_dict must be a dict or None")

        if vis_data_results is not None and not isinstance(vis_data_results, dict):
            raise TypeError("vis_data_results must be a dict or None")

        return result

    return wrapper


def get_cls(class_path: str, *args, **kwargs):
    """
    Dynamically import class and create instance.
    class_path format: 'mypkg.shapes.circle.Circle'
    """
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls

import os
import json

from tqdm.auto import tqdm


if __name__ == "__main__":
    target_path = "data/cleaned_aida"
    for filename in tqdm(os.listdir(target_path)):
        if filename == "batch_sampled_indices.json":
            continue

        image_id = filename
        assert os.path.exists(os.path.join(target_path, image_id, "raw_image.jpg")), image_id
        assert os.path.exists(os.path.join(target_path, image_id, "masked_image.png")), image_id
        assert os.path.exists(os.path.join(target_path, image_id, "metadata.json")), image_id

        with open(os.path.join(target_path, image_id, "metadata.json")) as f:
            metadata = json.load(f)

        for key in ["latex", "uuid", "unicode_str", "unicode_less_curlies", "image_data", "font", "filename"]:
            assert key in metadata

        for key in [
            "full_latex_chars", "visible_latex_chars", "visible_char_map", 
            "width", "height", "depth", "xmins", "xmaxs", "ymins", "ymaxs", "xmins_raw", "xmaxs_raw", "ymins_raw", "ymaxs_raw", "png_masks"
        ]:
            assert key in metadata['image_data']
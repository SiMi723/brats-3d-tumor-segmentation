def extract_patch(volume, size=128):
    
    x, y, z = volume.shape
    cx, cy, cz = x//2, y//2, z//2
    half = size // 2

    patch = volume[
        cx-half:cx+half,
        cy-half:cy+half,
        cz-half:cz+half
    ]

    return patch
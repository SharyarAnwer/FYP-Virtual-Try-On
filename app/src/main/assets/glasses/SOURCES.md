# 3D glasses models

Where each 3D glasses model in this folder came from. Every file here was written by
`tools/prepare_glasses.py` from the model as downloaded: parts found and renamed (Frame, Lenses,
ArmLeft, ArmRight), simplified to a triangle budget where needed, colours baked onto the corners,
scaled to real size - metres, +Y up, the front facing +Z - with the source's credit kept inside
the file. `catalog.json` lists the pairs the app offers; the tool writes it too.

The downloads themselves are not in the repository. To prepare a pair again, download it from its
source into `tools/glasses_sources/` (ignored by git), check its SHA-256 against the one below, and
run the command given.

| Pair (button name) | File | Source | Licence | Prepared |
|---|---|---|---|---|
| Aviators | `sunglasses_khronos.glb` | "SunglassesKhronos" from the Khronos Group's glTF Sample Assets, https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/SunglassesKhronos. Created by Eric Chadwick of Darmstadt Graphics Group GmbH, 2024. Khronos logo © 2015 Khronos Group; 3D Commerce logo © 2017 Khronos Group. | CC BY 4.0 International: free to use and change, with this credit | Not simplified (12,372 triangles; the inner lens surface left out); file already in metres, frame 149.8 mm wide |
| Black wood | `black_wood_frames.glb` | "Glasses" by Marius.Eder on Sketchfab, https://sketchfab.com/3d-models/glasses-007651a9450746a5b6c5a126d484cd52 | CC BY 4.0 International: free to use and change, with this credit | Simplified from 120,640 to 12,500 triangles; file in decimetres, frame 140.7 mm wide; wood texture baked onto the corners |
| Brown round | `brown_round_frames.glb` | "Glasses" by Ahmad Riazi on Sketchfab, https://sketchfab.com/3d-models/glasses-1620b0155c7b4948bdf71f2f8af390b9 | CC BY 4.0 International: free to use and change, with this credit | Frame and arms made as one piece, cut apart by shape; simplified from 31,016 to 12,402 triangles; unit unknown (105.5 mm wide by its own numbers, too narrow for its 17 mm bridge), scaled to 140 mm |
| Tortoise cat-eye | `tortoise_cat_eye.glb` | "Glasses (Rigel)" by Agha.Najam on Sketchfab, https://sketchfab.com/3d-models/glasses-rigel-47d044c57dc64586a09c0b181c34c874 | CC BY 4.0 International: free to use and change, with this credit | Viewer tilt of 3.1° straightened; simplified from 57,880 to 12,524 triangles; unit unknown (192 mm wide with 73 mm lenses as decimetres), scaled to 140 mm; metal and polish read from its metal-and-roughness picture; the tiny "CE" and "RIGEL" markings blur away when colours are baked |
| Black & violet | `black_violet_frames.glb` | "Glasses" by yanix on Sketchfab, https://sketchfab.com/3d-models/glasses-2a39a445c6b84ce89974ec2d0ae67061 | CC BY 4.0 International: free to use and change, with this credit | 3,968 fully transparent triangles dropped, then simplified from 114,520 to 12,498; file in metres, frame 135.9 mm wide; colours and see-through map baked onto the corners |

Downloads, SHA-256:

- `SunglassesKhronos.glb`: `25d72dd0869c99f94a3c0d6dfe0707714eca056e531f6ff77e7545719c86ab8c`
- `glasses.glb` (Sketchfab "Glasses" by Marius.Eder): `f04f0ebbaf52629a5afb971f7c8ca222f5c08158610804ffcdd0ab6b1f9dbb45`
- `glasses.glb` (Sketchfab "Glasses" by yanix): `553f6d029782cf4629384edc1d3da6924fb5c63eba0f670a8941fda19e9e5b74`
- `glasses.glb` (Sketchfab "Glasses" by Ahmad Riazi): `91844648f23b8d74aa1397bc33488490ab37a22f60402da170e170db1d893280`
- `glasses_rigel.glb` (Sketchfab "Glasses (Rigel)" by Agha.Najam): `b6341ded18bc9dd7e5eebf2f12d966a63ca6e54bf9360f300ca3e5cdaddf6bc1`

Commands, from the repository root:

    python tools/prepare_glasses.py tools/glasses_sources/sunglasses_khronos.glb --id sunglasses_khronos --name "Aviators" --frame-budget 6400 --arm-budget 2600
    python tools/prepare_glasses.py tools/glasses_sources/black_wood_frames.glb --id black_wood_frames --name "Black wood"
    python tools/prepare_glasses.py tools/glasses_sources/black_violet_frames.glb --id black_violet_frames --name "Black & violet"
    python tools/prepare_glasses.py tools/glasses_sources/brown_round_frames.glb --id brown_round_frames --name "Brown round"
    python tools/prepare_glasses.py tools/glasses_sources/tortoise_cat_eye.glb --id tortoise_cat_eye --name "Tortoise cat-eye"

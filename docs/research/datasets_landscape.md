# Datasets & Imagery Sources for Drone-Based Military Object Detection in Eastern & Southern Ukraine

## TL;DR
- **No single open dataset matches your exact need** (Russian military equipment, captured from drones at 300–800m oblique angles, over eastern/southern Ukrainian terrain, across seasons). The realistic path is to assemble: (a) several small open Roboflow Ukraine-war drone datasets for foreground objects, plus academic aerial benchmarks (DOTA, VEDAI, VisDrone, xView) for pretraining/augmentation, and (b) free **Sentinel-2 (Copernicus)** seasonal captures of the actual front-line oblasts for backgrounds, supplemented by VEDAI/LoveDA/DeepGlobe for higher-resolution rural texture.
- **The large "Ukraine combat" datasets are gated.** Enabled Intelligence's ~500,000-hour library, the Brave1 Dataroom, and the Ukraine MoD/Avengers "universal military dataset" are all restricted to approved US/NATO/Ukraine government and vetted-company users — none are publicly downloadable.
- **Geometry mismatch is the core technical caveat.** Satellite (nadir, 0.3–10m GSD) and most aerial benchmarks are top-down; your 300–800m oblique drone view has different parallax, shadow geometry, and object silhouettes. Treat satellite backgrounds as *texture/reference*; oblique drone/VisDrone-style imagery and your own captures are the geometrically appropriate plates for compositing.

## Key Findings
1. Openly downloadable Ukraine-war equipment imagery is concentrated in small Roboflow Universe community datasets (hundreds to ~1,200 images each, mostly CC BY 4.0, mixed quality), not in any large curated benchmark.
2. The richest real-combat datasets are access-controlled and require institutional/government partnership.
3. For backgrounds, the eastern/southern front (Donetsk, Luhansk, Zaporizhzhia, Kherson) is overwhelmingly open steppe cropland with very low forest cover (Kherson 4.8%, Donetsk ~7%, Luhansk ~11%) gridded by shelterbelts — a visually distinctive landscape that no off-the-shelf labeled dataset represents at drone-oblique angles.
4. Sentinel-2 is the single best free, legal, repeatable source for same-area summer/winter terrain of the actual front zones.
5. Seasonal change is dramatic (green/golden summer → black mud rasputitsa → snow/bare winter), so backgrounds must be pulled per-season.

## Details

### CATEGORY 1 — MILITARY / AERIAL OBJECT-DETECTION DATASETS

#### 1A. Roboflow Universe — Ukraine-war & military (FREE, mostly CC BY 4.0)
The most directly relevant *open* sources, though small and variable in quality. Pull via `pip install roboflow` then `roboflow.download_dataset(dataset_url="...", model_format="yolov8")`.

- **Military Vehicle Recognition** (MilitaryVehicleRecognition) — universe.roboflow.com/militaryvehiclerecognition/military-vehicle-recognition. ~1,195 images, 7 versions. Classes (5): tank, air-fighter, armoured personnel carrier, bomber, soldier. Described as labeled aerial images "captured by reconnaissance drones during the russo-Ukrainian War" at "diverse altitudes, angles, and lighting." License CC BY 4.0. Perspective: drone/aerial oblique-to-overhead. Caveat: described as imagery that "simulate[s] real-world conditions" — likely a mix of real and scraped frames. Reported model mAP@50 ~47%.
- **RussiantTankDroneImagesLowQuality** (user "tank") — universe.roboflow.com/tank-s4xwz/russianttankdroneimageslowquality. ~993 images. Classes include bmp-1, bmp-2, bmp-3, mt-lb, t-72. "All images from drones and low quality." CC BY 4.0. Perspective: low-altitude/FPV-style drone — the closest open match to FPV/low-oblique geometry.
- **military-vehicle-detection-juleg** (oleksandr-tara) — universe.roboflow.com/oleksandr-tara/military-vehicle-detection-juleg. License MIT.
- **Military Vehicle Detection** (DRGladwell) — universe.roboflow.com/drgladwell/military-vehicle-detection-yzlvf. 4K synthetic + real images aimed at spotting tanks at very long range (~32px). CC BY 4.0.
- **military object detection** (yolo datasets) — universe.roboflow.com/yolo-datasets-ymdve/military-object-detection-uxkcn. 2,636 images; classes: tank, military vehicle, soldier. CC BY 4.0.
- **Military Tanks** (MilitaryTanks) — universe.roboflow.com/militarytanks-c2etq/military-tanks. ~5,540 images; exports to all YOLO formats.
- Broader hits under universe.roboflow.com/search?q=class:military_vehicle show recurring Ukraine-war taxonomies (BMP, BMP-3, RSZO, SAU, MLRS, "orc-car," self-propelled howitzer, IFV) — the Ukrainian slang "orc" indicates war-sourced data. Generally CC BY 4.0, small.

#### 1B. Academic / benchmark aerial datasets (FREE for research)
None are drone-oblique over Ukraine, but they are essential for pretraining, transfer learning, and copy-paste augmentation.

- **DOTA (v1.0/v1.5/v2.0)** — Google Earth + JL-1/GF-2 satellite + some aerial. v1.0: 2,806 images, 188,282 instances, 15 classes; v1.5 adds tiny objects + container crane (16 classes, ~403K instances); v2.0: 11,268 images, ~1.79M instances, 18 classes (adds airport, helipad). GSD ~0.1–1m. Oriented bounding boxes. Relevant classes: small/large vehicle, plane, helicopter, storage tank, harbor. Near-nadir overhead.
- **xView** — WorldView-3 satellite, 0.3m GSD, ~1M objects, 60 classes, 846 images over 1,400 km². Includes trucks, passenger vehicles, maritime, building, storage tank. Strictly top-down. DIUx xView 2018 Challenge site (manual download).
- **VEDAI** — Vehicle Detection in Aerial Imagery; cropped from the Utah AGRC aerial survey (spring 2012). 1,246 images, 1024×1024 (12.5cm GSD) and 512×512 (25cm GSD), RGB + near-IR, 9–11 vehicle classes (car, truck, pickup, tractor, camping car, boat, van, plane...). Backgrounds: grass, highway, fields, urban — useful as rural texture. Near-nadir aerial.
- **VisDrone (VisDrone2019/2020-DET)** — drone-captured, 10,209 images + 288 video clips/261,908 frames, 10 classes (pedestrian, car, van, bus, truck, motor, bicycle, tricycle...), 14 Chinese cities, urban+rural. **Captured "from various shooting angles (vertical and oblique)" and variable altitude** — the closest mainstream benchmark to your oblique drone geometry, though objects are civilian and terrain is Chinese. Median object ~34px. AISKYEYE, Tianjin University.
- **DIOR** — 23,463 images, 800×800, 20 classes, GSD 0.5–30m. Horizontal + (DIOR-R) oriented boxes. Overhead.
- **FAIR1M** — >15,000 images, ~1M instances, 37 fine-grained classes (11 airplane, ship, vehicle subtypes). High-res satellite, oriented boxes.
- **MAR20** — largest open military *aircraft* recognition set: 3,842 images, 22,341 instances, 20 aircraft types, from 60 military airfields via Google Earth. HBB + OBB. gcheng-nwpu.github.io. Satellite overhead — relevant only if airfields/aircraft are in scope.
- **SODA-A** — small-object aerial subset, Google Earth, GSD ~0.5–0.8m, avg object ~14.75px, very dense tiny objects. Good for tiny-object training.
- **COWC (Cars Overhead With Context)** — overhead car detection/counting, ~15cm GSD, nadir. Good for small-vehicle pretraining.
- **ITCVD** — aerial vehicle detection (Netherlands), nadir (lower priority).
- **VETRA** (DLR) — vehicle tracking in aerial imagery, OBB/HBB, with more camera movement; newer benchmark worth checking.

#### 1C. Kaggle
- **Military Aircraft Detection Dataset** (a2015003713) — 100+ aircraft types, bounding boxes; ground/mixed perspectives. Also mirrored in YOLO format (rookieengg).
- **Military Assets Dataset (12 Classes, YOLO8)** (rawsi18) — 26,315 images, 12 classes including military_vehicle, soldier, military_artillery, trench, military_aircraft, military_warship. CC BY 4.0. Mixed perspective.
- **Camouflage Tank Detection** (ahmetfurkaann) — camouflaged tanks; useful for hard negatives/camo.
- **Military Aircraft Recognition / 81-class / 102-class** sets — classification-oriented.
- **2022 Ukraine-Russia War Equipment Losses + Oryx images** (piterfm) — destroyed-equipment photos (ground-level OSINT), not drone-perspective detection labels.

#### 1D. Gated Ukraine-war datasets (NOT publicly downloadable — listed for awareness)
- **Enabled Intelligence "EView" library** — per DefenseScoop (June 16, 2026), "half a million hours of Ukraine conflict drone footage" (EO, SAR, IR, foreign-language audio), described as "ready and available now" for approved users in the U.S., Ukraine and NATO nations; CEO Peter Kant: "What makes the Ukraine footage especially valuable is that it's real." Per DefenseScoop, "The National Geospatial-Intelligence Agency tapped Enabled Intelligence in 2025 for a single-award contract worth up to $708 million over a seven-year ordering period" (the Sequoia data-labeling-as-a-service contract, foundational to NGA's Maven program).
- **Brave1 Dataroom** (Ukraine MoD; built on Palantir Foundry, launched Jan 21, 2026) — per Ukraine MoD (June 12, 2026): "More than 100 Ukrainian companies have already gained access to Brave1 Dataroom and are using the platform to train AI models using real-world data." Defence Minister Mykhailo Fedorov: "The initial focus will be on autonomous detection and interception of aerial threats—a capability of critical importance to Ukraine." Secure environment; data never leaves it.
- **Avengers Labs / "universal military dataset"** (Ukraine MoD Delta project) — per Ukraine MoD, through Avengers Labs "Ukrainian and international companies can train their AI models on millions of annotated frames captured during real combat sorties." Train-in-place, gated.
- **Path to access:** institutional/government or vetted-defense-company partnership; pursue early if your project has official affiliation.

### CATEGORY 2 — TERRAIN / BACKGROUND IMAGERY OF EASTERN & SOUTHERN UKRAINE

#### 2A. Free satellite/aerial sources (the workhorses for backgrounds)
- **Sentinel-2 (Copernicus)** — THE recommended free source. Constellation of S-2A/B/C, 10m resolution (visible/NIR), 290km swath, 13 bands, archive back to 2015, free and open. Per ESA Sentinel Online, revisit is "10 days at the equator with one satellite, and 5 days with 2 satellites under cloud-free conditions which results in 2-3 days at mid-latitudes" — so you get frequent repeat coverage of Ukraine. Access via **Copernicus Browser** (browser.dataspace.copernicus.eu): pick your AOI (e.g., Pokrovsk, Zaporizhzhia front, Kherson), set one summer date range and one winter date range, filter to low cloud cover, and download **Level-2A** (surface reflectance, analysis-ready). This is how you get same-area summer vs winter/snow captures legally and repeatably. Perspective: nadir, 10m GSD (texture/reference only, NOT object-scale).
- **Landsat (USGS)** — 30m, free, long archive (large-area seasonal/historical context). Via USGS EarthExplorer (earthexplorer.usgs.gov) and NASA Earthdata.
- **Maxar Open Data Program** (now Vantor) — maxar.com/open-data. High-resolution (~0.3–0.5m) before/after imagery released for crisis events including Ukraine, **CC BY-NC 4.0**; also on AWS (registry.opendata.aws/maxar-open-data). Closest *free* high-res look at actual Ukraine terrain/events, but near-nadir satellite and event-driven (not systematic) coverage.
- **Planet Labs** — high-res (3–5m PlanetScope, ~0.5m SkySat) but **not free for Ukraine**: the free NICFI program covered only the tropics (30°N–30°S) and ended Jan 23, 2025; its successor (Tropical Forest Observatory) is also tropics-only. Route for Ukraine is Planet's **Education & Research Program** (go.planet.com/research) for qualifying students/researchers.
- **USGS EarthExplorer / OpenAerialMap** — see 2C.

#### 2B. Land-cover / semantic-segmentation datasets (for rural texture & negative backgrounds)
- **BigEarthNet** — 590,326 Sentinel-2 patches (1.2×1.2km), 43 CORINE classes (also a 19-class nomenclature), **across 10 European countries and all four seasons** (autumn 154,943 / winter 117,156 / spring 189,276 / summer 128,951 patches). Does NOT include Ukraine directly (Austria, Belgium, Finland, Ireland, Kosovo, Lithuania, Luxembourg, Portugal, Serbia, Switzerland), but Serbia/Lithuania/Kosovo terrain is reasonably analogous to Ukrainian agricultural/steppe-edge. 10/20/60m. Free.
- **LoveDA** — 5,987 images, 0.3m GSD, urban + rural domains, 7 classes (background, building, road, water, barren, forest, agriculture), from Chinese cities. **CC BY-NC-SA 4.0, academic only.** High-res rural agriculture/road/forest texture useful as reference.
- **DeepGlobe Land Cover** — 803–1,146 images, 2448×2448, 50cm GSD, 7 classes (urban, agriculture, rangeland, forest, water, barren, unknown), focused on rural areas. Good high-res rural texture.
- **Sen12MS** — 180,662 triplets (Sentinel-1 SAR + Sentinel-2 + MODIS land cover), global, all four seasons. Free. Good for multi-season, multi-modal context.
- **Dynamic World** — near-real-time 10m global land cover from Sentinel-2, 9 classes (water, trees, grass, crops, shrub/scrub, flooded veg, built, bare, snow/ice), via Google Earth Engine (`GOOGLE/DYNAMICWORLD/V1`), CC BY 4.0. **Best tool to characterize/segment the actual front-line land cover, derive field-vs-forest-vs-urban proportions for your AOI, and pick snow-covered vs snow-free dates.**
- **DynamicEarthNet** — daily Planet imagery, 75 global AOIs, monthly semantic labels — useful for seasonal-change modeling.
- **EuroSAT** — 27,000 Sentinel-2 patches, 10 classes, Europe — quick land-cover pretraining.
- **SpaceNet** — building/road extraction benchmarks (various cities); useful for urban/road structure, not Ukraine-specific.

#### 2C. Open drone/aerial repositories
- **OpenAerialMap** (openaerialmap.org / map.openaerialmap.org) — community repository of openly licensed drone + satellite imagery, **CC BY 4.0**, ~15,000+ images from satellite/drone/aircraft/balloon/kite. Searchable by area; metadata includes sensor, resolution, provider. Search for Ukraine or analogous Eastern-European agricultural captures — but coverage is sparse and not guaranteed for the front zones.
- **OpenDroneMap data / DroneMapper sample repositories** — sample drone datasets (agricultural fields, gravel pits, etc.), including oblique imagery, usable as generic rural backgrounds.
- **USGS EarthExplorer** — also hosts some UAV/aerial photography and LiDAR.

### CATEGORY 3 — GEOGRAPHY & TERRAIN OF THE EASTERN/SOUTHERN FRONT

The front-line oblasts sit in Ukraine's **steppe** (Pontic–Caspian) and **forest-steppe** belts, on deep, dark **chernozem** ("black earth") soils. This is the single most important fact for background realism: the terrain is overwhelmingly **open, flat-to-rolling cultivated cropland**, sparsely gridded by **shelterbelts / tree lines (лісосмуги/посадки)** — narrow planted barriers of trees protecting fields from wind and erosion, characteristic of the steppe and forest-steppe (the practice dates to early-19th-century Poltava and Dokuchaev's drought-control research).

Quantitatively (note: largely pre-2014 statistics):
- **Steppe zone nationally: 64.9% cultivated, only 4.2% woodland** (mostly riverbank + shelterbelt plantings), ~10.8% pasture (Internet Encyclopedia of Ukraine).
- **Forest-steppe zone: roughly two-thirds arable, ~one-eighth (≈12.5%) forest** (Encyclopædia Britannica).
- Per-oblast forest cover is far below Ukraine's ~16–17% national average: **Kherson 4.8% (least-forested oblast in Ukraine), Zaporizhzhia ~4% (approximate, regional source), Donetsk ~7% (forests+thickets+shelterbelts), Luhansk ~11%** (Internet Encyclopedia of Ukraine). Donetsk cropland is ~53.6% of total area.
- Arable land covers **70–90% of the southern and eastern regions** (multiple sources, incl. UWEC Work Group citing 2021 data; Wikipedia "Agriculture in Ukraine").

**Realistic background mix for the front zone:** dominant = **open agricultural field (поле)**; frequent = **shelterbelt/tree-line (посадка)** edges and **dirt roads (дорога)**; occasional = **forest (ліс)** only along the Donets Ridge, Serebryansky Forest (Luhansk), and pine terraces of the Siverskyi Donets; plus **urban/peri-urban (місто)** and **Donbas mining/industrial terrain** (spoil tips, slag heaps); plus the **Dnipro river system** in the south. **Mud (багнюка)** is a seasonal *state* of fields and unpaved roads, not a separate land cover.

**Seasonal visual change is dramatic from above:**
- **Summer:** green/golden fields (wheat, sunflower, corn), harvest patterns, dust on dry dirt roads, full green tree lines. High contrast, strong shadows.
- **Autumn (osennaya rasputitsa / бездоріжжя):** the mud season — heavy October rains turn chernozem fields and unpaved roads into deep black/brown mud; vehicles confined to predictable hard-road corridors; tree lines turning/bare. Chernozem's clay content makes the mud uniquely deep and adhesive.
- **Winter:** snow cover (variable, often partial/patchy in the south), bare dark tree lines starkly visible against snow or bare black soil, low contrast, low sun angle / long shadows, frozen hard ground.
- **Spring (vesnyanaya rasputitsa):** snowmelt + thaw = mud again, then rapid green-up.

Because the same field can be green, golden, black-mud, snow-white, or bare-brown within a single year, **you must pull backgrounds per-season** — a summer-only background set will not generalize to winter operations.

### CATEGORY 4 — PERSPECTIVE / GEOMETRY MISMATCH (critical for compositing)

Three distinct geometries, NOT interchangeable:
- **Satellite nadir** (Sentinel-2 10m; xView/Maxar 0.3–0.5m; DOTA/DIOR/MAR20/SODA-A): straight-down, near-parallel view, shadows determined by sun only, minimal parallax, objects seen as roofs/tops. GSD coarse (Sentinel-2) to fine (Maxar).
- **High-altitude / nadir aerial** (VEDAI 12.5–25cm, COWC, DeepGlobe 50cm, LoveDA 30cm): also essentially top-down but higher resolution; vehicles still seen mostly from above.
- **Drone oblique at 300–800m (your case)**: the camera looks *down and forward at an angle*, so you see vehicle **sides + tops**, get **foreshortening and parallax**, tree lines/buildings cast **long oblique shadows and occlude ground objects**, and the horizon or distant background may appear. **VisDrone** (vertical + oblique, variable altitude) and **SeaDronesSee** (5–260m altitude, 0–90° gimbal pitch, with altitude/angle metadata) are the only mainstream benchmarks that capture this oblique geometry — though not your objects or terrain.

**Implications for using satellite backgrounds in a drone-detection pipeline:**
- Nadir satellite/aerial imagery is **geometrically wrong** as a literal background plate for an oblique drone scene — ground-plane perspective, object shadows, and parallax won't match, so naive compositing produces "floating," mis-scaled, or mis-shadowed objects that can *degrade* a detector (a documented failure mode in copy-paste literature; random/ill-matched paste can lower mAP rather than raise it).
- **Geometrically appropriate** background sources for compositing onto oblique synthetic data: VisDrone oblique frames, your own/borrowed drone captures at matching altitude/angle, and OpenAerialMap oblique drone imagery — these share the ground-plane perspective.
- **Useful only as texture/reference** (color palette, crop patterns, shelterbelt spacing, seasonal tone, snow appearance): Sentinel-2, Maxar, DeepGlobe, LoveDA, VEDAI. Use these to build seasonal color/texture references, drive a perspective warp/homography before compositing, or train a separate terrain/background classifier — not as literal oblique plates.
- **Best practice for your synthetic pipeline:** extract foreground vehicles (from the Roboflow Ukraine sets / VisDrone-angle captures), match scale to ~300–800m oblique GSD, apply a perspective transform to any nadir-derived background, match sun/shadow direction and season, and use **context- and depth-aware copy-paste** (e.g., Dvornik context model, Simple Copy-Paste, depth-aware compositing) rather than random paste to avoid the documented degradation from geometric/illumination mismatch.

## Recommendations
**Stage 1 — Foreground objects (do first):**
- Pull all the open Roboflow Ukraine-war sets (Military Vehicle Recognition, RussiantTankDroneImagesLowQuality, military-vehicle-detection-juleg, Military Tanks) and the Kaggle Military Assets 12-class set. Deduplicate and standardize to YOLO format — these give you real Russian-equipment instances at drone-ish angles.
- Pretrain/transfer from **VisDrone** (for oblique geometry) and **DOTA/VEDAI** (for overhead vehicle features); add **SODA-A/COWC** for tiny-object robustness.

**Stage 2 — Backgrounds (in parallel):**
- Use **Sentinel-2 via Copernicus Browser** to pull summer AND winter Level-2A captures of your specific AOIs (Pokrovsk, Avdiivka/Donetsk front, Zaporizhzhia line, Kherson/Dnipro). Use **Dynamic World** (Earth Engine) to confirm land-cover proportions and to locate snow-free vs snow dates and field/forest/shelterbelt mixes.
- Add high-res rural texture from **VEDAI, DeepGlobe, LoveDA** for shelterbelt/field/road appearance.
- Search **OpenAerialMap** for oblique drone captures over Ukraine or analogous Serbian/Romanian/southern-Russian steppe.

**Stage 3 — Synthetic compositing:**
- Composite extracted vehicles onto perspective-warped, season-matched backgrounds using context/depth-aware copy-paste; match shadow direction and GSD. Hold out a validation set of real frames (camouflaged/netted/partially hidden vehicles), as the Ukrainian Avengers team does for false-positive benchmarking.

**Thresholds that change the plan:**
- If you obtain **institutional access to the Brave1 Dataroom or Enabled Intelligence**, deprioritize synthetic compositing — real combat full-motion video will dominate. Pursue official partnership early.
- If detector recall on **winter/snow** scenes lags summer meaningfully in validation, pull more winter Sentinel-2 dates and synthesize more snow/mud backgrounds.
- If false positives cluster on **shelterbelts/tree-line shadows**, mine more shelterbelt negative backgrounds specifically.

## Caveats
- The biggest honest gap: **there is no open dataset of Ukrainian eastern/southern terrain at drone-oblique angles across seasons.** You must construct it from Sentinel-2 seasonal pulls + generic European agricultural/oblique drone imagery + your own captures.
- Roboflow Ukraine sets are **small and of mixed provenance** ("simulate real-world conditions" — not all verified combat frames); treat labels and realism critically.
- **License watch:** LoveDA and MAR20 are academic/non-commercial only; **Maxar Open Data is CC BY-NC** (no commercial use); several Roboflow sets are CC BY 4.0 (attribution) while one (juleg) is MIT. Check each against your project's commercial status.
- Per-oblast forest/land-cover figures derive largely from pre-2014 statistics, and the Zaporizhzhia ~4% figure is from a tourism-grade source (flagged approximate); wartime landscape change (e.g., Serebryansky Forest destruction, cratering, abandoned overgrown fields) post-dates them.
- The Enabled Intelligence figures (~500,000 hours; up to $708M NGA contract) trace to company statements via DefenseScoop (June 2026) — a single originating source.
- Geometry mismatch is real and quantified in the literature: naive use of nadir backgrounds can *reduce* detector performance. Treat satellite imagery as texture/reference, not literal oblique plates.
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from skimage.metrics import structural_similarity as ssim

def task2_clustering_improved(image_path, k=10, spatial_weight=0.05, exg_weight=1.5, texture_weight=1.2):
    print(f"--- Виконання кластеризації (k={k}) ---")
    img = cv2.imread(image_path)
    if img is None:
        print(f"Помилка: не знайдено {image_path}")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 1. Фільтрація
    denoised = cv2.bilateralFilter(img_rgb, d=12, sigmaColor=40, sigmaSpace=75)
    img_lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)

    h, w, _ = img_lab.shape

    # 2. Просторові координати
    x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))

    # 3. Розрахунок ExG (вегетаційний індекс)
    R, G, B_chan = cv2.split(denoised.astype(np.float32))
    exg = (2 * G - R - B_chan)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mu = cv2.blur(gray, (5, 5))
    mu2 = cv2.blur(gray ** 2, (5, 5))
    texture = np.sqrt(np.maximum(mu2 - mu ** 2, 0))

    # Формуємо масив з 7 ознак: L, A, B, X, Y, ExG, Texture
    pixel_features = np.zeros((h * w, 7), dtype=np.float32)
    pixel_features[:, :3] = img_lab.reshape((-1, 3))
    pixel_features[:, 3] = x_coords.flatten()
    pixel_features[:, 4] = y_coords.flatten()
    pixel_features[:, 5] = exg.flatten()
    pixel_features[:, 6] = texture.flatten()

    scaler = MinMaxScaler()
    pixel_features_scaled = scaler.fit_transform(pixel_features)
    pixel_features_scaled[:, 3] *= spatial_weight
    pixel_features_scaled[:, 4] *= spatial_weight
    pixel_features_scaled[:, 5] *= exg_weight
    pixel_features_scaled[:, 6] *= texture_weight

    # Кластеризація KMeans
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixel_features_scaled)

    # Відновлення кольорів кластерів 
    centers_rgb = np.zeros((k, 3), dtype=np.uint8)
    pixels_rgb_flat = img_rgb.reshape((-1, 3))

    for i in range(k):
        mask = (labels == i)
        if np.any(mask):
            centers_rgb[i] = np.mean(pixels_rgb_flat[mask], axis=0)

    segmented_image = centers_rgb[labels].reshape(img_rgb.shape)

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].imshow(img_rgb)
    ax[0].set_title("Оригінал")
    ax[1].imshow(denoised)
    ax[1].set_title("Фільтрація (Bilateral)")
    ax[2].imshow(segmented_image)
    ax[2].set_title(f"Segmented (k={k})")

    for a in ax:
        a.axis('off')

    plt.tight_layout()
    plt.show()

    return segmented_image

def task3_counting(image_path):
    print("--- Група 3: Гібридний підхід (Adaptive + Watershed) ---")
    img = cv2.imread(image_path)
    if img is None:
        print("Помилка: Зображення не знайдено.")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_copy = img_rgb.copy()

    # 1. Попередня обробка
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # 2. Адаптивна бінаризація
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 21, 4)

    # 3. Морфологічне очищення
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)
    sure_bg = cv2.dilate(closing, kernel, iterations=2)

    # 4. Центри об'єктів (Distance Transform)
    dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # 5. Підготовка до Watershed
    unknown = cv2.subtract(sure_bg, sure_fg)

    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(img, markers)
    count = 0
    min_area = 90

    for marker_id in np.unique(markers):
        if marker_id <= 1:
            continue

        mask = np.zeros(gray.shape, dtype="uint8")
        mask[markers == marker_id] = 255

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            cnt = max(contours, key=cv2.contourArea)
            if cv2.contourArea(cnt) > min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
                count += 1

    print(f"Знайдено об'єктів: {count}")

    # Візуалізація
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    ax[0].imshow(img_rgb)
    ax[0].set_title("Оригінал")

    ax[1].imshow(dist_transform, cmap='jet')
    ax[1].set_title("Карта відстаней (Distance Transform)")

    ax[2].imshow(img_copy)
    ax[2].set_title(f"Результат: {count} об'єктів")

    for a in ax: a.axis('off')
    plt.tight_layout()
    plt.show()

def task4_comparison(img1_path, img2_path):
    print("---  Група 4: Порівняння ---")
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1 is None or img2 is None:
        print("Помилка: не знайдено одне з зображень для порівняння.")
        return

    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(5000)
    keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    good_matches = matches[:int(len(matches) * 0.15)]

    if len(good_matches) > 10:
        src_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        img2_aligned = cv2.warpPerspective(img2, M, (img1.shape[1], img1.shape[0]))
        gray2_aligned = cv2.cvtColor(img2_aligned, cv2.COLOR_BGR2GRAY)

        valid_mask = np.ones_like(gray2) * 255
        warped_mask = cv2.warpPerspective(valid_mask, M, (img1.shape[1], img1.shape[0]))
    else:
        print("Попередження: Не знайдено достатньо точок для вирівнювання.")
        img2_aligned = img2
        gray2_aligned = gray2
        warped_mask = np.ones_like(gray2) * 255

    (score, diff) = ssim(gray1, gray2_aligned, full=True)
    diff = (diff * 255).astype("uint8")

    _, thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    mask_erosion_kernel = np.ones((5, 5), np.uint8)
    warped_mask = cv2.erode(warped_mask, mask_erosion_kernel, iterations=2)
    thresh = cv2.bitwise_and(thresh, thresh, mask=warped_mask)

    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    dilated = cv2.dilate(cleaned, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_aligned_rgb = cv2.cvtColor(img2_aligned, cv2.COLOR_BGR2RGB)
    result_img = img2_aligned_rgb.copy()

    for cnt in contours:
        if cv2.contourArea(cnt) > 200:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(result_img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    ax[0, 0].imshow(img1_rgb)
    ax[0, 0].set_title("Зображення 1 (База)")
    ax[0, 1].imshow(img2_aligned_rgb)
    ax[0, 1].set_title("Зображення 2 (Вирівняне)")

    ax[1, 0].imshow(diff, cmap='gray')
    ax[1, 0].set_title("Мапа різниці (SSIM)")

    ax[1, 1].imshow(result_img)
    ax[1, 1].set_title("Знайдені відмінності")

    for a in ax.flat:
        a.axis('off')

    plt.suptitle("Група 4: Порівняння зображень (з маскуванням країв)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    img_for_cluster = "img_4.png"
    img_for_count = "img_1.png"
    img_to_compare_1 = "img_2.png"
    img_to_compare_2 = "img_3.png"

    task2_clustering_improved(img_for_cluster, k=10)
    task3_counting(img_for_count)
    task4_comparison(img_to_compare_1, img_to_compare_2)
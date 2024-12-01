from django.shortcuts import render
import os
from django.conf import settings
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import proc_engine
import numpy as np

def index(request):
    return render(request, 'index.html')

def checker(test_image_path1, test_image_path2):
    image_1 = proc_engine.testing(test_image_path1)
    image_2 = proc_engine.testing(test_image_path2)

    data_array_1 = np.array(image_1, dtype=np.float64)
    data_array_2 = np.array(image_2, dtype=np.float64)

    # Print extracted features
    print("------------Image 1 Features-----", data_array_1)
    print("------------Image 2 Features-----", data_array_2)

    # Normalize feature vectors
    data_array_1 /= np.linalg.norm(data_array_1)
    data_array_2 /= np.linalg.norm(data_array_2)

    # Calculate Euclidean distance and Cosine similarity
    euclidean_distance = np.linalg.norm(data_array_1 - data_array_2)
    cosine_similarity = np.dot(data_array_1, data_array_2)

    # Print calculated metrics
    print("Euclidean Distance:", euclidean_distance)
    print("Cosine Similarity:", cosine_similarity)

    # Define thresholds for matching
    euclidean_threshold = 0.7  # Higher threshold for allowing small differences
    cosine_threshold = 0.75      # Higher threshold for similarity

    # Determine if the images match
    if euclidean_distance < euclidean_threshold and cosine_similarity > cosine_threshold:
        print("Images match.")
        return True
    else:
        print("Images do not match.")
        return False
    
@csrf_exempt
def compare_sigature(request):
    fileInput1 = request.FILES.get('fileInput1')
    fileInput2 = request.FILES.get('fileInput2')
    temp_dir = os.path.join(settings.MEDIA_ROOT, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    temp_path_1 = os.path.join(temp_dir, fileInput1.name)
    temp_path_2 = os.path.join(temp_dir, fileInput2.name)
    with default_storage.open(temp_path_1, 'wb') as destination:
        for chunk in fileInput1.chunks():
            destination.write(chunk)
    with default_storage.open(temp_path_2, 'wb') as destination:
        for chunk in fileInput2.chunks():
            destination.write(chunk)
    check = checker(temp_path_1, temp_path_2)
    if os.path.exists(temp_path_1):
        os.remove(temp_path_1)
    if os.path.exists(temp_path_2):
        os.remove(temp_path_2)
    return JsonResponse({'status':1, 'engine_res':check, })
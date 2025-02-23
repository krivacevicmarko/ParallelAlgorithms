# -*- coding: utf-8 -*-
"""run.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1UB1TM9pPutCxQHw-8tnqOOBOTciAVzTa
"""

import sys
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import time
import threading
import json
import multiprocessing
import os
from queue import Queue

images_registry = {}
task_registry = {}
register_locks = {}
image_counter = 0
task_counter = 0

exit_signal = False
condition = threading.Condition()
lock = threading.Lock()
output_dir = "/content/slike/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
images_register_locks = {}
task_registry_lock = threading.Lock()
image_counter_lock = threading.Lock()
image_conditions = {}
message_queue = Queue()
task_queue = Queue()

def addImageToRegistry(image_path, task_id=None, output_path=None):
  global image_counter
  global images_registry

  if not os.path.exists(image_path):
      sys.stderr.write(f"Greška: Slika na putanji {image_path} ne postoji.\n")
      return
  try:
      image_array = load_image(image_path)
  except Exception as e:
      sys.stderr.write(f"Greška pri učitavanju slike sa putanje {image_path}: {str(e)}\n")
      return
  with image_counter_lock:
    image_counter += 1
    image_id = image_counter

  images_register_locks[image_id] = threading.Lock()
  images_register_locks[image_id].acquire()

  if output_path is None:
    output_path = f"{output_dir}{image_id}.png"

  image = Image.fromarray(image_array)
  image.save(output_path)

  images_registry[image_id] = {
      "image_id": image_id,
      "image_array": image_array,
      "image_path": output_path,
      "from_task": [],
      "from_image": [],
      "used_in_tasks": [],
      "delete_flag": False,
      "filters_applied": [],
      "processing_time": None,
      "size": image_array.shape
  }
  images_register_locks[image_id].release()
  sys.stdout.write(f"Slika {image_id} dodata u registar i direktorijum slike\n")

def markForDeletion(image_id):
  if image_id in images_registry:
    images_registry[image_id]["delete_flag"] = True

def addTaskToRegistry(image_id, task_id, filter_name, parameters):
  global task_registry
  with task_registry_lock:
    task_registry[task_id] = {
        "task_id": task_id,
        "image_id": image_id,
        "filter_name": filter_name,
        "parameters": parameters,
        "status": "spremna za obradu"
    }
  return task_id

def load_image(image_path):
  image = Image.open(image_path)
  return np.array(image)

def load_JSON_file(json_path):
    with open(json_path) as f:
        params = json.load(f)
    return params

def adjust_brightness(image_array, factor=1.0):
  mean_intensity = np.mean(image_array, axis=(0, 1), keepdims=True)
  image_array = (image_array - mean_intensity) * factor + mean_intensity
  adjusted_image = np.clip(image_array, 0, 255)
  return adjusted_image.astype(np.uint8)

def gaussian_blur(image_array, sigma=1):
  red_channel = gaussian_filter(image_array[..., 0], sigma=sigma)
  green_channel = gaussian_filter(image_array[..., 1], sigma=sigma)
  blue_channel = gaussian_filter(image_array[..., 2], sigma=sigma)
  blurred_image = np.zeros_like(image_array)
  blurred_image[..., 0] = red_channel
  blurred_image[..., 1] = green_channel
  blurred_image[..., 2] = blue_channel

  if image_array.shape[-1] == 4:
      alpha_channel = image_array[..., 3]
      blurred_image[..., 3] = alpha_channel

  blurred_image = np.clip(blurred_image, 0, 255)
  return blurred_image.astype(np.uint8)

def grayscale(image_array):
  red_channel = image_array[..., 0]
  green_channel = image_array[..., 1]
  blue_channel = image_array[..., 2]
  grayscale_image = (red_channel * 0.299 + green_channel * 0.587 + blue_channel * 0.114)
  return grayscale_image.astype(np.uint8)

def list_images():
  messages = []
  for image_id,image_data in images_registry.items():
    message = f"Image ID {image_id}, Path: {image_data['image_path']}"
    messages.append(message)
  message_queue.put(f"Images in registry:\n {messages}")

def describe_image(image_id):
  image_data = images_registry.get(int(image_id), None)
  if image_data:
    message_queue.put(f"Image ID: {image_data['image_id']}, FromTask: {image_data['from_task']} , FromImage: {image_data['from_image']}\n")
  else:
    sys.stdout.write(f"Nije pronađena slika sa ID {image_id}\n")

def message_controller(interval=10):
  next_check = time.time() + interval
  while True:
    try:
      current_time = time.time()
      if current_time >= next_check:
        next_check = current_time + interval
        if message_queue.empty():
          sys.stdout.write("Nema poruka u redu za poruke\n")
        else:
          while not message_queue.empty():
            message = message_queue.get()
            if message == "STOP":
              sys.stdout.write("Zaustavljanje message_controller-a\n")
              return
            sys.stdout.write(f"Poruka iz queue-a za poruke:\n{message}\n")
            time.sleep(0.1)
    except Exception as e:
      sys.stdout.write(f"Greška u message_controller: {e}\n")

def check_dependencies(image_id, current_task_id):
    global task_registry
    # Ako slika još nije dostupna (koristi se u tasku za kreiranje)
    if image_id not in images_registry:
        sys.stdout.write(f"Zadatak {current_task_id} čeka na kreiranje slike sa ID {image_id}...\n")
        if image_id not in image_conditions:
            image_conditions[image_id] = threading.Condition()

        # Sačekaj dok slika ne postane dostupna
        with image_conditions[image_id]:
            while image_id not in images_registry:
                image_conditions[image_id].wait()
        sys.stdout.write(f"Slika sa ID {image_id} je kreirana, zadatak {current_task_id} nastavlja obradu.\n")

    # Provera da li postoje aktivni zadaci koji koriste istu ulaznu sliku
    if image_id in image_conditions:
        with image_conditions[image_id]:
            for task_id, task_data in task_registry.items():
                if task_data["image_id"] == image_id and task_data["status"] != "završeno" and task_id != current_task_id:
                    sys.stdout.write(f"Zadatak {current_task_id} čeka na završetak zadatka {task_id} (zavisnost slike ID {image_id})...\n")
                    task_registry[task_id]["status"] = "ceka na obradu"
                    image_conditions[image_id].wait()
                    sys.stdout.write(f"Zadatak {task_id} je završen, zadatak {current_task_id} nastavlja obradu.\n")

def addNewImageToRegistry(output_path, image_id, new_image_id, task_id, image_array, transformation,processing_time):
  global images_registry
  if new_image_id not in images_register_locks:
     images_register_locks[new_image_id] = threading.Lock()
  images_register_locks[new_image_id].acquire()
  images_registry[new_image_id] = {
      "image_id": new_image_id,
      "image_array": image_array,
      "image_path": output_path,
      "from_task": images_registry[image_id]["from_task"] + [task_id],
      "from_image": images_registry[image_id]["from_image"] + [image_id],
      "used_in_tasks": [],
      "delete_flag": False,
      "filters_applied": images_registry[image_id]["filters_applied"] + [transformation],
      "processing_time": processing_time,
      "size": image_array.shape
  }
  images_registry[image_id]["used_in_tasks"].append(task_id)
  images_register_locks[new_image_id].release()

def update_registry(result):
  output_path,image_id,new_image_id, task_id, new_image_array, transformation , processing_time  = result
  global task_registry
  with image_counter_lock:
    if new_image_id in images_registry:
        sys.stderr.write(f"Upozorenje: Pokušan unos slike sa već postojećim ID {new_image_id}!\n")
        return
  addNewImageToRegistry(output_path,image_id,new_image_id, task_id, new_image_array, transformation , processing_time)
  task_queue.put((task_id,image_id,new_image_id))

  with task_registry_lock:
    task_registry[task_id]["status"] = "završeno"

def process_image(task_id, image_id, image_path, new_image_id,task_registry,images_registry,output_path):
  transformation = task_registry[task_id]['filter_name']
  params = task_registry[task_id]['parameters']

  start_time = time.time()

  if transformation == "gaussian_blur":
    sigma = params.get("sigma",1)
    new_image_array = gaussian_blur(images_registry[image_id]['image_array'],sigma)

  elif transformation == "adjust_brightness":
    factor = params.get("factor",1.0)
    new_image_array = adjust_brightness(images_registry[image_id]['image_array'],factor)

  elif transformation == "grayscale":
    new_image_array = grayscale(images_registry[image_id]['image_array'])

  else:
    raise ValueError("Unknown transformation")

  end_time = time.time()
  processing_time = end_time - start_time
  time.sleep(15)
  new_image = Image.fromarray(new_image_array)
  new_image.save(output_path)
  sys.stdout.write(f"Sacuvana slika {new_image_id}\n")

  return output_path,image_id,new_image_id, task_id, new_image_array, transformation , processing_time


def handleprocess(json,output_path=None):
  global image_counter
  global task_registry
  global images_registry
  global task_counter
  transformation_data = load_JSON_file(json)
  image_id = transformation_data.get("image_id")
  transformation_type = transformation_data.get('transformation')
  parameters = transformation_data.get('parameters',{})

  if image_id in images_registry:
    if images_registry[image_id]['delete_flag'] == True:
      sys.stdout.write(f"Ne mozete pristupiti slici {image_id},jer je oznacena za brisanje\n")
      return

  task_counter += 1
  task_id = task_counter

  addTaskToRegistry(image_id, task_id, transformation_type, parameters)

  check_dependencies(image_id, task_id)

  image_path = images_registry[image_id]['image_path']

  with image_counter_lock:
    image_counter += 1
    new_image_id = image_counter

  check_dependencies(image_id, task_id)

  with task_registry_lock:
    task_registry[task_id]["status"] = "u obradi"
    sys.stdout.write(f"Zadatak {task_id} zapoceo obradu. \n")

  if output_path is None:
    output_path = f"{output_dir}{new_image_id}.png"
  pool.apply_async(process_image,args=(task_id,image_id,image_path,new_image_id,task_registry,images_registry,output_path),callback=update_registry)

def delete_image_from_registry(image_id):
  global images_registry, task_registry

  if image_id not in images_registry:
    sys.stdout.write(f"Slika ID {image_id} ne postoji u registru.\n")
    return

  images_register_locks[image_id].acquire()

  try:
    active_tasks = [task_id for task_id, task_data in task_registry.items()
                    if task_data["image_id"] == image_id and task_data["status"] != "završeno"]

    if active_tasks:
      sys.stdout.write(f"Slika ID {image_id} se koristi u aktivnim zadacima {active_tasks}. Čekanje na završetak...\n")

      for task_id in active_tasks:
        while task_registry[task_id]["status"] != "završeno":
          time.sleep(0.001)

        sys.stdout.write(f"Svi zadaci za sliku ID {image_id} su završeni. Nastavljam sa brisanjem...\n")

    if images_registry[image_id]["delete_flag"] == True:
      image_data = images_registry.pop(image_id)
      sys.stdout.write(f"Slika ID {image_id} uspešno obrisana iz registra.\n")
    else:
      sys.stdout.write(f"Slika ID {image_id} nije označena za brisanje.\n")
  finally:
    images_register_locks[image_id].release()

def update_task_queue():
  while True:
    try:
      task_data = task_queue.get()
      if task_data == "STOP":
        sys.stdout.write("Zaustavljanje update_task_queue niti\n")
        break

      task_id, image_id, new_image_id = task_data
      if new_image_id in image_conditions:
        with image_conditions[new_image_id]:
          image_conditions[new_image_id].notify_all()
      sys.stdout.write(f"Završena obrada za task {task_id}, nova slika dodata u registar sa ID {new_image_id}\n")

    except Exception as e:
      sys.stdout.write(f"Greška u update_task_queue: {e}\n")
      break

def shutdown_checker():
  global exit_signal, threads
  while not exit_signal:
    time.sleep(1)

  sys.stdout.write("Signal za gašenje primljen. Čekanje na završetak aktivnih niti...\n")

  for thread in threads:
    thread.join()
  message_queue.put("STOP")
  task_queue.put("STOP")
  task_thread.join()
  message_thread.join()

def process_command(command):
  parts = command.split()
  if len(parts) == 0:
    return

  cmd = parts[0]

  if cmd == "add":
    global image_counter
    image_path = parts[1]
    t = threading.Thread(target=addImageToRegistry,args=(image_path,))
    t.start()
    threads.append(t)

  elif cmd == "process":
    json_file_path = parts[1]
    t = threading.Thread(target=handleprocess,args=(json_file_path,))
    t.start()
    threads.append(t)

  elif cmd == "list":
    t = threading.Thread(target=list_images)
    t.start()
    threads.append(t)

  elif cmd == "describe":
    image_id = parts[1]
    t = threading.Thread(target=describe_image,args=(image_id,))
    t.start()
    threads.append(t)

  elif cmd == "delete":
    image_id = int(parts[1])
    markForDeletion(image_id)
    sys.stdout.write(f"Slika ID {image_id} je oznacena za brisanje\n")
    t = threading.Thread(target=delete_image_from_registry,args=(image_id,))
    t.start()
    threads.append(t)

  elif cmd == "tasks":
    sys.stdout.write(f"Taskovi u registru:{task_registry}\np")
    sys.stdout.write(f"Slike u registru:{images_registry}\np")


  elif cmd == "exit":
    global exit_signal
    exit_signal = True
    return False

  else:
    sys.stdout.write(f"Unknown command: {cmd}")

  return True

if __name__ == "__main__":
  running = True
  cpu = multiprocessing.cpu_count()
  pool = multiprocessing.Pool(cpu)
  task_thread = threading.Thread(target=update_task_queue)
  task_thread.start()
  message_thread = threading.Thread(target=message_controller,daemon=True)
  message_thread.start()
  shutdown_thread = threading.Thread(target=shutdown_checker,daemon=True)
  shutdown_thread.start()

  while running:
    global image_counter
    global task_counter
    threads = []

    command = input("> ")
    running = process_command(command)

  shutdown_thread.join()
  pool.close()
  pool.join()
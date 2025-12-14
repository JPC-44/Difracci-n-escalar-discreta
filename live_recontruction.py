#  pip install imagingcontrol4
#  pip install imagingcontrol4pyside6
# https://www.theimagingsource.com/en-us/documentation/ic4python/api-reference.html#module-imagingcontrol4.queuesink
import imagingcontrol4 as ic4
import sys
def x():
    ic4.Library.init()

    grabber = ic4.Grabber()

    # abrir y habilitar la cámara
    first_device_info = ic4.DeviceEnum.devices()[0]
    grabber.device_open(first_device_info)

    # resolucion
    grabber.device_property_map.set_value(ic4.PropId.WIDTH, 4000)
    grabber.device_property_map.set_value(ic4.PropId.HEIGHT, 3000)
    exposure_property = grabber.device_property_map.get_property(ic4.PropId.EXPOSURE_TIME)
    grabber.device_property_map.set_value(ic4.PropId.EXPOSURE_TIME, 3000)

    print(exposure_property.unit)
    grabber.device_property_map._get_prop_handle
    # habilita obtener una sola imagen o secuencia de imagenes fuera de "data stream"
    sink = ic4.SnapSink()



    grabber.stream_setup(sink, setup_option=ic4.StreamSetupOption.ACQUISITION_START)

    try:
        image = sink.snap_single(time_out = 1000)
        #image = sink.snap_sequence(1000)
        print(f"received an image. ImageType:{image.image_type}")

        image.save_as_png("imagingsource.png")

    except ic4.IC4Exception as ex:
        print(ex.message)

    grabber.stream_stop()
    grabber.device_close()



def example_save_jpeg_file():

    # cantidad de dispositivos (cámaras )
    device_list = ic4.DeviceEnum.devices()
    print(device_list)
    #sys.exit(0)
    # listado por característica y asignación de índice
    for i, dev in enumerate(device_list):
        print(f"[{i}] {dev.model_name} ({dev.serial}) [{dev.interface.display_name}]")
    print(f"Select device [0..{len(device_list) - 1}]: ", end="")

    # selección de índice
    selected_index = int(input())
    dev_info = device_list[selected_index]

    # Abrir el dispositivo selecciionado en un nuevo grabber
    grabber = ic4.Grabber(dev_info)

    # Crear un snapsink para captura manual en buffer 
    sink = ic4.SnapSink()

    # Inicia el stream data 
    grabber.stream_setup(sink)

    for i in range(10):
        input("Press ENTER to snap and save a jpeg image")

        # Grab the next image buffer
        buffer = sink.snap_single(1000)

        # Save buffer contents in a jpeg file
        filename = f"image_{i}.jpeg"
        buffer.save_as_jpeg(filename, quality_pct=100)

        print(f"Saved image file {filename}")
        print()

    # Only for completeness. Technically this is not necessary here, since the grabber is destroyed at the end of the function.
    grabber.stream_stop()
    grabber.device_close()



def example_save_jpeg_file_():
    device_list = ic4.DeviceEnum.devices()
    for i, dev in enumerate(device_list):
        print(f"[{i}] {dev.model_name} ({dev.serial}) [{dev.interface.display_name}]")
    print(f"Select device [0..{len(device_list) - 1}]: ", end="")
    selected_index = int(input())
    dev_info = device_list[selected_index]

    # Open the selected device in a new Grabber
    grabber = ic4.Grabber(dev_info)

    # Create a snap sink for manual buffer capture
    sink = ic4.SnapSink()

    # Start data stream from device to sink
    grabber.stream_setup(sink)

    for i in range(10):
        input("Press ENTER to snap and save a jpeg image")

        # Grab the next image buffer
        buffer = sink.snap_single(1000)

        # Save buffer contents in a jpeg file
        filename = f"image_{i}.jpeg"
        buffer.save_as_jpeg(filename, quality_pct=90)

        print(f"Saved image file {filename}")
        print()

    # Only for completeness. Technically this is not necessary here, since the grabber is destroyed at the end of the function.
    grabber.stream_stop()
    grabber.device_close()

x()
import numpy as np
import sys

from pygrabber.dshow_graph import client, clsids, ICreateDevEnum, GUID, POINTER, IPropertyBag, qedit, wstring_at, DeviceCategories
from pygrabber.moniker import IMoniker

def get_input_devices(prop):
        return _get_available_filters(DeviceCategories.VideoInputDevice, prop)

def _get_available_filters(category_clsid, prop):
    system_device_enum = client.CreateObject(clsids.CLSID_SystemDeviceEnum, interface=ICreateDevEnum)
    filter_enumerator = system_device_enum.CreateClassEnumerator(GUID(category_clsid), dwFlags=0)
    moniker, count = filter_enumerator.Next(1)
    result = []
    while count > 0:
        result.append(_get_filter_name(moniker, prop))
        moniker, count = filter_enumerator.Next(1)
    return result


def _get_filter_name(arg, prop="DevicePath"):
    if type(arg) == POINTER(IMoniker):
        property_bag = arg.BindToStorage(0, 0, IPropertyBag._iid_).QueryInterface(IPropertyBag)
        return property_bag.Read(prop, pErrorLog=None)
    elif type(arg) == POINTER(qedit.IBaseFilter):
        filter_info = arg.QueryFilterInfo()
        return wstring_at(filter_info.achName)
    else:
        return None


# from https://stackoverflow.com/questions/49546179/python-normalize-image-exposure

def get_histogram(img):
  '''
  calculate the normalized histogram of an image
  '''
  height, width = img.shape
  hist = [0.0] * 256
  for i in range(height):
    for j in range(width):
      hist[img[i, j]]+=1
  return np.array(hist)/(height*width)

def get_cumulative_sums(hist):
  '''
  find the cumulative sum of a numpy array
  '''
  return [sum(hist[:i+1]) for i in range(len(hist))]

def normalize_histogram(img):
  # calculate the image histogram
  hist = get_histogram(img)
  # get the cumulative distribution function
  cdf = np.array(get_cumulative_sums(hist))
  # determine the normalization values for each unit of the cdf
  sk = np.uint8(255 * cdf)
  # normalize the normalization values
  height, width = img.shape
  Y = np.zeros_like(img)
  for i in range(0, height):
    for j in range(0, width):
      Y[i, j] = sk[img[i, j]]
  # optionally, get the new histogram for comparison
  new_hist = get_histogram(Y)
  # return the transformed image
  return Y
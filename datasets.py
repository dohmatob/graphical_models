import os
import urllib2
import numpy as np


def load_data(url, web=True, data_dir=None, redownload=False):
    """
    Grabs data from local or remote URL.

    Parameters
    ----------
    url: string
        local filename or remote URL (on the web) containing data to grab

    data_dir: string, optional (default None)
        local directory supposedly containing the requested data. If this
        directory exists and contains the url basename as a file, then the
        data will be read from this file; else data is downloaded over the
        web as usual, and then stored as a file (with same basename as the url)
        under data_dir.

    redownload: boolean, optional (default False)
        if set, then the data will be downloaded, even if it alread exists
        locally (a kind of update operation)

    Returns
    -------
    data: numpy array
        the downloaded data

    Throws
    ------
    URLError if url is broken.

    """

    if not web:  # read from local file
        data = np.loadtxt(url)
    else:  # grab from web
        if data_dir is None:
            data_dir = os.getcwd()
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        data_filename = os.path.join(data_dir, os.path.basename(url))
        if not redownload and os.path.isfile(data_filename):
            print "%s exists; reading data from local file ..." % (
                data_filename)
            data = np.loadtxt(data_filename)
        else:
            print "Openning %s ..." % url
            raw_data = urllib2.urlopen(url).read().replace('\t', ' ')
            data = np.array([np.fromstring(row, sep=' ')
                             for row in raw_data.split('\n')[:-1]])
            np.savetxt(data_filename, data, fmt='%f')

    # return loaded data
    print "... done."
    return data

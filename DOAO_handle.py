#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
created on Wednesday Feb. 07 17:31:00, 2022

@author: D.H. Hyun

#------------------------------------------------------------
- version 1.1.6 -

IMPORTABLE FILE of functions for creating IMSNG reference images
>> import DOAO_handle as dh (You see the connection to the author?)

purposed for DOAO data, but still adaptable to other contexts
with minor revisions including file paths & selection criteria

This file consists of two parts.
Part 1 is about choosing the right images to combine.
Part 2 is for aligning and stacking the chosen images,
    and other miscellaneous options and after-processing.

You get ~5s speed boost per 'refim' when you turn 'report' off.
#------------------------------------------------------------
If you intend to customize this code for your own usage, 
    the followings are where you should look first.

* Change the path of the data
    - 'path' in 'tns_select', 'pool' in 'select'
      (for 'tns_select', go also see the docstring there)
    - 'fieldlist' & 'filterlist' in 'USAGE EXAMPLE'

* Check if the Keys match the ones in the headers of your data
    - 'SEEING' and 'UL5_2' in 'imqual_select'
    - 'SKYVAL' in 'align'

* Change the pre-obtained figures
    - 'median_seeing','median_depth','std_seeing','std_depth'
       in 'imqual_select'

* Check the selection criteria
    - 'sp', 'dp' in 'imqual_select'
    - 'thresh', 'imqual', 'pixcut' in 'select'
    - 'least' in 'refim'

* Change the name of the result & the path it is saved to.
    - 'infolist' in 'refim'
    - 'filename', 'dirpath' in 'saveas'
    -  all 'fits.writeto()' in 'refim'

* Check the header updating format
    - 'COMATHR', 'IMCOM{}', 'NCOMBINE' in 'update_hdr'

* Check the stacking method
    - 'combine_ccddata' and 'combine_numpy' in 'refim'
       and its arguments, 'bottleneck' and 'nan'
       
Also, refer to the USAGE EXAMPLE at the end of the file.
#------------------------------------------------------------
*Overview of the Algorithm

refim():
    select():
        tns_select()
        imqual_select()
        align_select()
    
    align():
        astroalign
    
    combine_ccddata():
        ccdproc Combiner with bn_median
    
    crop()
    
    fits.writeto()
#------------------------------------------------------------
'start' and 'cp' variables scattered all over the place are 
for measuring the running time for each snippet of code.

You can conveniently add or delete a checkpoint using
'cp = lapse(report, previous cp, explanation)'.

Here, 'report' is a boolean value of which default is True.
All 'report' is controlled at once as a keyword argument in 'refim'.
#------------------------------------------------------------
*Packages to Install
- numpy
- astropy
- astroalign ('pip install' recommended)
- ccdproc
- bottleneck
#------------------------------------------------------------
*REVISION LOG

version 1.0 -> 1.0.1
    - minor revisions.
version 1.0.1 -> 1.1
    - image cropping utilities added
    - bug fixes for the image selecting algorithm not working 
      properly when fed with too little data
version 1.1 -> 1.1.1
    - bug fixes for astroalign facing MaxIterError with bad images
      & hdrlist not being converted to hdr_array properly
version 1.1.1->1.1.2
    - trimmed out some spaghetti code (bug fix for 'flow')
version 1.1.2->1.1.3
    - header update utility added
    - selection flow optimization
    - now the target frame uses Best-SEEING idx in lack of data
version 1.1.3->1.1.4
    - bug fix for calculating idx_seeing before bool_array
version 1.1.4->1.1.5
    - now combine info. is appended to the bottom of the hdr.
version 1.1.5->1.1.6
    - minor revisions on file paths for the author's new accout
    - os.path used for Windows environment
    - now 'flow' fits in the 60-char. print format
"""
#              ===========
# = = = =======   PART 1  ======= = = =
#              ===========

import numpy as np
from timeit import default_timer as timer
from astropy.io import fits


# -----------------------------------------------------------------------------
def get_date(current, value, opt="month"):
    from datetime import datetime
    from dateutil.relativedelta import relativedelta

    info = current.split("-")
    year = int(info[0])
    month = int(info[1])
    day = int(info[2])
    if opt == "month":
        delta = relativedelta(months=value)
    elif opt == "day":
        delta = relativedelta(days=value)
    dt = datetime(year, month, day) + delta
    new_date = "".join(str(dt.date()).split("-"))
    return new_date


# -----------------------------------------------------------------------------
def tns_select(field, imlist):
    """
    It rules out the images including transients
    based on the Transient Name Server.
    The csv files in the right format must be prepared in advance.
    >> ask hhchoi1022 for the TNS API searching code

    In case you don't figure out how to get this code working,
    you can manually remove the images from the folder where
    this code retrieves its data. Then simply disable tns_select by
    commenting out the block, 'imlist = tns_select(field,imlist)'.
    """
    from astropy.table import Table

    try:
        path = f"/data3/hhchoi1022/TNS_DOAO/TNS_{field}.csv"
        dates = Table.read(path)["DetDate"]
        ts = Table()
        starts, ends = [], []
        for line in dates:
            if line != "None":  # 'none'
                date = line.split(" ")[0]
                start = get_date(date, -10, opt="day")
                end = get_date(date, 6, opt="month")
                starts.append(start)
                ends.append(end)
        ts.add_column(starts, name="start")
        ts.add_column(ends, name="end")
    except FileNotFoundError as err:
        print("ABORT:", err)

    datelist = [int(img.split("-")[-4]) for img in imlist]
    new_imlist = []
    for n, date in enumerate(datelist):
        flag = False
        for line in ts:
            if date >= int(line[0]) and date <= int(line[1]):
                flag = True
        if flag == False:
            new_imlist.append(imlist[n])
    return new_imlist


# -----------------------------------------------------------------------------
"""
Run the following lines to get SEEING/DEPTH cuts.
This will take tens of minutes depending on the amount of data.

from astropy.io import fits
import glob
allfile = '/data3/hhchoi1022/IMSNG/IMSNGgalaxies/*/*/*/*60.fits'
total_imlist = glob.glob(allfile)

total_seeing = []
total_depth= []
for n, image in enumerate(total_imlist):
    try:
        hdr = fits.getheader(image)
        total_depth.append(hdr['UL5_2'])
        total_seeing.append(hdr['SEEING'])
    except KeyError as err:
        print(f'total_imlist index={n}',err)
median_seeing,std_seeing= np.median(total_seeing),np.std(total_seeing)
median_depth, std_depth = np.median(total_depth), np.std(total_depth)
"""


# -----------------------------------------------------------------------------
def imqual_select(hdrlist, step, detail=False):
    seeing = []
    depth = []
    err_index_list = []
    n = 0
    for hdr in hdrlist:
        seeing.append(hdr["SEEING"])
        # for exceptional cases with no 'UL5_2'
        try:
            depth.append(hdr["UL5_2"])
        except:
            # nan values always give False comparing arrays
            depth.append(np.NaN)
            print(f"requested key missing. index={n}\n")  # ,imlist[n])
            err_index_list.append(n)
        n += 1

    # SEEING/DEPTH statistics were previously obtained. Refer to above
    median_seeing, std_seeing = 3.201, 0.8079601996720359
    median_depth, std_depth = 19.288, 0.6618451009359285

    sp = [-0.7, 0, 0.7, 3]
    dp = [0.5, 0, -0.5, -3]

    cut_seeing = round(median_seeing + sp[step] * std_seeing, 2)
    cut_depth = round(median_depth + dp[step] * std_depth, 3)

    bool_seeing = np.array(seeing) <= cut_seeing
    bool_depth = np.array(depth) >= cut_depth
    bool_array = bool_seeing & bool_depth

    if err_index_list != None:
        bool_array[err_index_list] = False

    if not detail:
        return bool_array
    elif detail:
        seeing = [s for s, v in zip(seeing, bool_array) if v]
        return bool_array, cut_seeing, cut_depth, seeing


# -----------------------------------------------------------------------------
def align_select(hdr_sel, pixel=150):
    from astropy.wcs import WCS

    # importing FITSFixedWarning to ignore the datfix: MJD-OBS warning
    from astropy.wcs import FITSFixedWarning
    import warnings

    warnings.simplefilter("ignore", category=FITSFixedWarning)

    # extract wcs & get image dimensions
    wcslist = []
    shape = (2048, 2048)
    for n, hdr in enumerate(hdr_sel):
        _shape_ = (hdr["NAXIS1"], hdr["NAXIS2"])
        wcslist.append(WCS(hdr))
        if n > 0 and shape != _shape_:
            print("\tWARNING: Inconsistent Dimensions", shape, _shape_)
        shape = _shape_

    # use pixel_to_world for the midpoint
    sky = wcslist[0].pixel_to_world(shape[0] // 2, shape[1] // 2)

    # use world_to_pixel for all the other images
    pixlist = []
    for w in wcslist:
        coord = w.world_to_pixel(sky)
        pixlist.append((np.round(coord[0]), np.round(coord[1])))

    x = [coord[0] for coord in pixlist]
    y = [coord[1] for coord in pixlist]
    xm, ym = np.median(x), np.median(y)

    aligncut1 = x <= xm + pixel
    aligncut2 = x >= xm - pixel
    aligncut3 = y <= ym + pixel
    aligncut4 = y >= ym - pixel
    aligncut = aligncut1 & aligncut2 & aligncut3 & aligncut4

    return aligncut, pixlist, (xm, ym)


# -----------------------------------------------------------------------------
def choose_ref(pixlist_sel, mid=None):
    """
    It determines the center image based on the given list of pixel-
    coordinates representing a point in the sky of the same RA, Dec.

    The midpoint of the given list of coordinates can be specified in
    order to avoid repeating calculations already done in align_select().
    """
    if mid == None:
        x = [coord[0] for coord in pixlist_sel]
        y = [coord[1] for coord in pixlist_sel]
        mid = (np.median(x), np.median(y))
    dist = [np.linalg.norm(np.array(pair) - mid) for pair in pixlist_sel]
    return dist.index(min(dist))


# -----------------------------------------------------------------------------
def lapse(report, cp1, exp, end="\n"):
    """
    lapse takes a predefined timeit.default_timer value
    and returns the current timeit.default_timer value,
    printing the difference of the two with an explanation.
    """
    # timeit.default_timer was imported as timer() beforehand
    cp2 = timer()
    if report:
        print(f" {cp2-cp1:.3f}s {exp}", end=end)
    return cp2


# -----------------------------------------------------------------------------
def select(field, filtertype, report=True, detail=False, start=0):
    """
    takes 'field', and 'filtertype' as string inputs
    returns a list of filenames

    If detail==False, select() returns a numpy array of
    filenames of selected images.
    Otherwise, it additionally returns a list of headers of the
    selected images and an index of the center image.
    """
    import glob

    # Running Message
    if report:
        wrd = "=" * 21 + " SELECTING IMAGES " + "=" * 21
        print(f"\n{wrd}")

    # checkpoint for timer. no print if report==False
    if start == 0:
        cp = timer()
    else:
        cp = lapse(report, start, "importing packages & defining userfuncs")

    # least number of images to combine. alterable
    thresh = 9

    # retrieve the list of fits files
    pool = (
        "/data3/hhchoi1022/IMSNG/IMSNGgalaxies/"
        + field
        + "/DOAO/"
        + filtertype
        + "/Calib-DOAO-*-20??*-*-*-60.fits"
    )
    imlist = sorted(glob.glob(pool))
    if not imlist:  # implicit booleanness. pythonic:)
        raise FileNotFoundError("ABORT: no such file in the given directory")

    #     cp = lapse(report,cp,'for glob')

    # exclude images containing transients
    imlist = tns_select(field, imlist)

    cp = lapse(report, cp, "for TNS selection")

    if not imlist:
        raise FileNotFoundError("ABORT: nothing left after TNS sifting")

    # load headers from imlist
    hdrlist = [fits.getheader(img) for img in imlist]

    cp = lapse(report, cp, "constructing hdrlist")

    # selection criteria
    imqual = ["BEST", "NORMAL", "BAD", "WORST"]
    pixcut = [150, 200]

    terminate = False
    nlist = []
    for i in range(4):
        # selection from seeing/depth
        bool_array, cut_seeing, cut_depth, seeing = imqual_select(
            hdrlist, i, detail=True
        )

        # construct new lists of img names and hdrs
        im_array = np.array(imlist)[bool_array]
        """CAUTION: Converting a list of hdrs into an ndarray
                    occasionally leads to an error. Keep it a list.
           If you still want to do so, specify [dtype=object]"""
        hdr_list = [hdr for hdr, entry in zip(hdrlist, bool_array) if entry]
        n_sel = len(im_array)

        cp = lapse(report, cp, "for image quality")

        # selection from misalignment
        pixnum = len(pixcut)
        for j in range(pixnum):
            if n_sel == 0:
                pass
            else:
                aligncut, pixlist, mid = align_select(hdr_list, pixel=pixcut[j])
                im_sel = im_array[aligncut]
                n_sel = len(im_sel)
            nlist.append(n_sel)

            cp = lapse(report, cp, f"for misalignment ({pixcut[j]}pix)")

            if n_sel >= thresh:
                terminate = True
                break
        if terminate:
            break

    # raise an Error to stop further execution
    if n_sel < thresh:
        err = f"ABORT: only {n_sel} (<{thresh}) images left after selection"
        raise FileNotFoundError(err)

    # print the information
    if report:
        print("\nSelection Flow:")
        flow = []
        for m, num in enumerate(nlist):
            imqq = imqual[m // pixnum]
            pixx = pixcut[m % pixnum]
            flow.append(f"{imqq}, {pixx}pix: {num} survived")
        flow = " >> ".join(flow)
        for ite in range(len(flow) // 60 + (1 if len(flow) % 60 != 0 else 0)):
            print(flow[60 * ite : 60 * (ite + 1)])
        print(f"{'-'*60}")
        status = (
            "#\t{} frames, {} quality criteria met.\n"
            + '#\tSEEING cut = {}", DEPTH cut = {} mag'
        )
        print(status.format(n_sel, imqual[i], cut_seeing, cut_depth))
        print(f"#\tmisalignment selection of {pixcut[j]} pixels")
        print(f"{'-'*60}")
        print("Pass [report=False] to deactivate the status report.\n")

    # return early if you only want the selected imlist
    if not detail:
        _ = lapse(report, start, "for total select()")
        return im_sel

    # select an image to be the reference frame
    """
    outside of align_select for optimization purposes.
    In lack of data, the ref. frame is switched to the best-seeing image to 
    prevent AstroAlign having a bad image as the target to align others with.
    """
    if i == 3:  # and n_sel < 1.5*thresh:
        seeing = [s for s, v in zip(seeing, aligncut) if v]
        idx = seeing.index(min(seeing))
        print("Switched the ref. frame from the middle to the Best-SEEING.\n")
    else:
        pix_sel = np.array(pixlist)[aligncut]
        idx = choose_ref(pix_sel, mid)

    hdr_sel = [hdr for hdr, entry in zip(hdr_list, aligncut) if entry]

    _ = lapse(report, start, "for total select()")
    return im_sel, hdr_sel, idx


#              ===========
# = = = =======   PART 2  ======= = = =
#              ===========


def align(im_sel, idx=0, headers=[], report=True, start=0):
    """
    align() aligns the given images using ASTROALIGN module.
    It takes a list of names(including filepath) of the images,
    and will choose the first image of the list as the reference
    (target of astroalign) of alignment.

    You can designate the index of the ref. image for your own purpose
    through the keyword arg. 'idx', and pass a list of the headers of
    the images to save time.
    """
    import astroalign as aa

    # Running Message
    if report:
        wrd = "=" * 21 + " ALIGNING IMAGES " + "=" * 22
        print(f"{wrd}")

    # running time checkpoint
    if start == 0:
        cp = timer()
    else:
        cp = lapse(report, start, "initiating align()")
    if report:
        print(" aligning images...", end="\r")

    # construct hdrlist in case it is not given
    if len(headers) == 0:
        headers = []
        for img in im_sel:
            headers.append(fits.getheader(img))

    path_ref = im_sel[idx]
    data_ref = fits.getdata(path_ref)
    hdr_ref = headers[idx]
    sky_ref = hdr_ref["SKYVAL"]

    # initialize a 2048 x 2048 False array
    bool_array = np.zeros(np.shape(data_ref), dtype=bool)
    new_im_sel = [path_ref]
    data_aligned = [data_ref]

    AAError = False

    # align the source_images except the target_image i == idx
    for i in [j for j in range(len(im_sel)) if j != idx]:
        data = fits.getdata(im_sel[i])
        hdr = headers[i]
        # scale the background flux w.r.t. the reference
        data_scaled = data - hdr["SKYVAL"] + sky_ref
        """
        In case of Big/Little Endian Mismatch Error, try
        data = data.byteswap().newbyteorder()
        """
        # astroalign may exhaust its triangles before finding a transformation
        try:
            registered_image, footprint = aa.register(
                data_scaled, data_ref, fill_value=np.NaN
            )

            bool_array = bool_array | footprint
            data_aligned.append(registered_image)
            new_im_sel.append(im_sel[i])

        except:
            AAError = True
            name = im_sel[i].split("/")[-1]
            print(f"AstroAlign failed for a bad image: {name}")
            # raise FileNotFoundError(
            #     'ABORT: astroalign failed due to bad images')

    if AAError:
        print("\n[CAUTION]: Excluded bad images during AstroAlign")
        print("The result may not be of acceptable quality\n")

    cp = lapse(report, cp, "for astroalign")

    boolmap = 1 - bool_array.astype("int")
    return new_im_sel, data_aligned, hdr_ref, boolmap


# -----------------------------------------------------------------------------
"""
You can choose the method of imcombine among the following four options.

_numpy is the fastest, returning nan values for less stacked regions.
- If you want to be returned the values rather than np.NaN, pass [nan=False].

_ccddata incorporates the utility of excluding outliers, being way slower.
- It imports Bottleneck to accelerate the median_combine.
- Pass [bottleneck=False] to disable bn_median().

_ccddata is recommended for a better quality of result.
"""


# -----------------------------------------------------------------------------
def combine_numpy(data_aligned, nan=True, report=True, start=0):
    """
    'data_aligned' is a python list whose each entry is an image
    in the form of a 2D numpy ndarray.

    If you pass nan=False, combine_numpy will use np.nanmedian
    instead of np.median, giving a float value for each pixel whose
    values in all stacked images include np.NaN.
    """

    # Running Message
    if report:
        wrd = "=" * 21 + " STACKING IMAGES " + "=" * 22
        print(f"\n{wrd}")

    # checkpoint
    if start == 0:
        cp = timer()
    else:
        cp = lapse(report, start, "initiating combine_numpy()")
    if report:
        print(" combining images...", end="\r")

    if not nan:
        data_stacked = np.nanmedian(data_aligned, axis=0)
        cp = lapse(report, cp, "for numpy median")
    else:
        data_stacked = np.median(data_aligned, axis=0)
        cp = lapse(report, cp, "for numpy nanmedian")

    return data_stacked


# -----------------------------------------------------------------------------
def bn_median(masked_array, axis=None):
    """
    for details, refer to
    https://ccdproc.readthedocs.io/en/v0.2/ccdproc/bottleneck_example.html
    """
    import numpy as np
    import bottleneck as bn

    data = masked_array.filled(fill_value=np.NaN)
    med = bn.nanmedian(data, axis=axis)
    return np.ma.array(med, mask=np.isnan(med))


# -----------------------------------------------------------------------------
def combine_ccddata(data_aligned, bottleneck=True, report=True, start=0):
    """
    'data_aligned' is a python list whose each entry is an image
    in the form of a 2D numpy ndarray.

    If you pass bottleneck=False, combine_ccddate will avoid using the
    Bottleneck package, which is advantageous in case you have a problem
    installing it. However, the combining procedure may take longer.
    """
    from astropy.nddata import CCDData
    from ccdproc import Combiner

    # Running Message
    if report:
        wrd = "=" * 21 + " STACKING IMAGES " + "=" * 22
        print(f"{wrd}")

    # checkpoint
    if start == 0:
        cp = timer()
    else:
        cp = lapse(report, start, "initiating combine_ccddata()")
    if report:
        print(" combining images...", end="\r")

    aligned_imlist = CCDData(data_aligned, unit="adu")
    combiner = Combiner(aligned_imlist, dtype=np.float32)

    if bottleneck:
        comdata = combiner.median_combine(median_func=bn_median)
        cp = lapse(report, cp, "for ccdproc Combiner (bottleneck)")

    elif not bottleneck:
        comdata = combiner.median_combine()
        cp = lapse(report, cp, "for ccdproc Combiner")

    return comdata.data


# -----------------------------------------------------------------------------
def cropcoord(boolmap):
    y, x = np.shape(boolmap)
    xpix, ypix = 0, 0

    trim = True
    while trim:
        sumlist = []
        sumlist.append(sum(boolmap[0, :]) / x)
        sumlist.append(sum(boolmap[:, 0]) / y)
        sumlist.append(sum(boolmap[-1, :]) / x)
        sumlist.append(sum(boolmap[:, -1]) / y)
        idx = sumlist.index(min(sumlist))

        if idx == 0:
            boolmap = boolmap[1:, :]
            ypix += 1
        elif idx == 1:
            boolmap = boolmap[:, 1:]
            xpix += 1
        elif idx == 2:
            boolmap = boolmap[:-1, :]
        elif idx == 3:
            boolmap = boolmap[:, :-1]
        y, x = np.shape(boolmap)
        # This condition is exact, taking the 4 corners into consideration
        if sum(sumlist) == 4:
            trim = False
    # WATCH the position of x & y!
    return (xpix + x / 2, ypix + y / 2), (y, x)


# -----------------------------------------------------------------------------
def crop(position, size, data, hdr):
    """
    CAUTION: All warnings here is ignored to avoid getting
    repetitive messages for non-standard wcs input. So be aware
    you may not be informed of some important warnings.
    """
    from astropy.nddata import Cutout2D
    from astropy.wcs import WCS

    import warnings

    warnings.simplefilter("ignore")

    wcs = WCS(hdr, relax=False)
    cutout = Cutout2D(data, position=position, size=size, wcs=wcs)
    hdr.update(cutout.wcs.to_header())
    return cutout.data, hdr


# -----------------------------------------------------------------------------
def update_hdr(hdr, imlist, ncombine):
    import os

    hdr.append(("COMATHR", "Donghwan Hyun"), bottom=True)
    for i in range(ncombine):
        hdr.append((f"IMCOM{i}", os.path.basename(imlist[i])), bottom=True)
    hdr.append(("NCOMBINE", ncombine), bottom=True)
    return hdr


# -----------------------------------------------------------------------------
def saveas(infolist):
    import os

    filename = "-".join(infolist)
    dirhere = os.path.dirname(__file__)
    dirpath = os.path.join(dirhere, "IMSNG", "reference")
    # '/data3/dhhyun0223/IMSNG/Reference/'
    if not os.path.isdir(dirpath):
        os.system(f"mkdir {dirpath}")  # mkdir works in Windows too!
    filename = os.path.join(dirpath, filename)
    return filename


# -----------------------------------------------------------------------------
def refim(field, filtertype, report=True, least=6):
    """
    CAUTION: You MUST use this function in the manner of
    try:
        refim('NGC1234','V')
    except FileNotFoundError as err:
        print(err)
    .
    This whole set of functions incorporates the specific type of Error
    as an indicator in the selection process.
    You may change the type of the error like ctrl+shift+f, FileNotFoundError
        -> ZeroDivisionError for the sake of debugging or etc.

    If you define start = timer() before this function, you can assess
    the time taken to initiate the functions.
    """
    # Checkpoint to measure the total time
    start = timer()

    if report:
        print("\n+" + "-" * 58 + "+")
        wrd = "Creating Ref. Image for"
        print(f"{'-'*12} {wrd} {field}-{filtertype} {'-'*13}")
        print("+" + "-" * 58 + "+")

    # Select
    im_sel, hdr_sel, idx = select(
        field, filtertype, detail=True, report=report, start=start
    )
    cp = timer()

    # Align
    im_sel, data_aligned, hdr_ref, boolmap = align(
        im_sel, idx, headers=hdr_sel, report=report, start=cp
    )
    cp = timer()

    # raise error if the aligned images < the least cut
    """This is necessary since we might lose images during AstroAlign 
    in the WORST case. The cut here 'least', the maximum number of images
    per day in DOAO, is different from 'thresh' above, 
    which I arbitrarily set to 9 to allow some leeway."""
    n_sel = len(im_sel)
    if n_sel < least:
        wrd = "not enough left after AstroAlign"
        raise FileNotFoundError(f"ABORT: {wrd}: {n_sel} (<{least})")

    # Stack
    #     data_stacked = combine_numpy(data_aligned,report=report,start=cp)
    data_stacked = combine_ccddata(data_aligned, report=report, start=cp)

    # append imcombine info. to the header
    hdr_ref = update_hdr(hdr_ref, im_sel, n_sel)

    ## Change the Filename & etc. Here ##

    # total exposure
    exposure_per_image = 60
    exposure_total = str(exposure_per_image * n_sel)

    # filename
    infolist = ["Ref", "DOAO", field, filtertype, exposure_total]
    corename = saveas(infolist)

    # write new .fits files with the combined data
    fits.writeto(f"{corename}.raw.fits", data_stacked, hdr_ref, overwrite=True)
    fits.writeto(f"{corename}.map.fits", boolmap, overwrite=True)

    # trim edges of the image where the stacking was incomplete
    position, size = cropcoord(boolmap)
    data_cut, hdr_cut = crop(position, size, data_stacked, hdr_ref)

    fits.writeto(f"{corename}.com.fits", data_cut, hdr_cut, overwrite=True)

    end = timer()
    if report:
        print(f" {end-start:.3f}s total for {field}-{filtertype}\n")
        print("COMPLETE!\n")


# -----------------------------------------------------------------------------

# If you import this file, the test code below will not be executed
if __name__ == "__main__":
    refim("NGC4303", "R")

"""
                    ===============
        ============ USAGE EXAMPLE ============
                    ===============

import os
# import random
from timeit import default_timer as timer
import DOAO_handle as dh
# if 'legacy' folder contains a __init__.py file.
# import legacy.DOAO_handle_1_1_1 as dh


fieldlist = sorted(os.listdir('/data3/hhchoi1022/IMSNG/IMSNGgalaxies/'))
# fieldlist = random.sample(fieldlist,10)
filterlist = ['B','V','R']


start = timer()

done=0
nofile=0
unexpected=0

for field in fieldlist:
    for filtertype in filterlist:
        try:
            dh.refim(field,filtertype)
            done+=1
        except FileNotFoundError as err:
            print(err,'\n\n')
            nofile+=1
        except:
            print(f'Something Else has Gone Wrong; {field}-{filtertype}\n\n')
            unexpected+=1

end = timer()

print(f'total {len(fieldlist)*len(filterlist)}, well done {done}, '\
        + f'FileNotFound/NotEnoughData {nofile}, something else {unexpected}')
print(f'{end-start}s for {len(fieldlist)} fields')
"""

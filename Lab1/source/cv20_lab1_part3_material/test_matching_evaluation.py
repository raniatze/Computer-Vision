# Assuming detectors are in file "cv20_lab1_part2.py", replace with your filename.
import cv20_lab1_part3_utils as p3
import cv20_lab1_part2 as p2
import cv2
import numpy as np

if __name__ == '__main__':

    Detectors = ['AngleDetect', 'MultiScalarAngleDetect', 'BlobsDetect', 'MultiScalarBlobsDetect', 'MultiScalarBoxFilter']
    detect_fun = [0]*5
    desc_fun = [0]*10

    # Here is a lambda which acts as a wrapper for detector function, e.g. harrisDetector.
    # The detector arguments are, in order: image, sigma, rho, k, threshold.
    detect_fun[0] = lambda Ι: p2.AngleDetect(Ι,2,2,0.012,0.01)[0]
    desc_fun[0] = lambda Ι, kp: p3.featuresSURF(Ι, kp)
    desc_fun[1] = lambda I, kp: p3.featuresHOG(I,kp)

    detect_fun[1] = lambda Ι: p2.MultiScalarAngleDetect(Ι,2,2.5,1.8,4,0.01,0.007)
    desc_fun[2] = lambda Ι, kp: p3.featuresSURF(Ι, kp)
    desc_fun[3] = lambda I, kp: p3.featuresHOG(I,kp)

    detect_fun[2] = lambda Ι: p2.BlobsDetect(Ι,1.5,0.08)[0]
    desc_fun[4] = lambda Ι, kp: p3.featuresSURF(Ι, kp)
    desc_fun[5] = lambda I, kp: p3.featuresHOG(I,kp)

    detect_fun[3] = lambda Ι: p2.MultiScalarBlobsDetect(Ι,2,1.5,4,0.16)
    desc_fun[6] = lambda Ι, kp: p3.featuresSURF(Ι, kp)
    desc_fun[7] = lambda I, kp: p3.featuresHOG(I,kp)

    detect_fun[4] = lambda Ι: p2.MultiScalarBoxFilter(Ι,2,1.5,4,0.12)
    desc_fun[8] = lambda Ι, kp: p3.featuresSURF(Ι, kp)
    desc_fun[9] = lambda I, kp: p3.featuresHOG(I,kp)

    for i,detector in enumerate(Detectors):

        # Execute evaluation by providing the above functions as arguments
        # Returns 2 1x3 arrays containing the errors
        avg_scale_errors_surf, avg_theta_errors_surf = p3.matching_evaluation(detect_fun[i], desc_fun[2*i])
        avg_scale_errors_hog, avg_theta_errors_hog = p3.matching_evaluation(detect_fun[i], desc_fun[2*i+1])

        print('Avg. Scale Error for Image 1 with {} and SURF descriptor: {:.3f}'.format(detector,avg_scale_errors_surf[0]))
        print('Avg. Scale Error for Image 1 with {} and HOG  descriptor: {:.3f}'.format(detector,avg_scale_errors_hog[0]))

        print('Avg. Theta Error for Image 1 with {} and SURF descriptor: {:.3f}'.format(detector,avg_theta_errors_surf[0]))
        print('Avg. Theta Error for Image 1 with {} and HOG  descriptor: {:.3f}'.format(detector,avg_theta_errors_hog[0]))

        print('Avg. Scale Error for Image 2 with {} and SURF descriptor: {:.3f}'.format(detector,avg_scale_errors_surf[1]))
        print('Avg. Scale Error for Image 2 with {} and HOG  descriptor: {:.3f}'.format(detector,avg_scale_errors_hog[1]))

        print('Avg. Theta Error for Image 2 with {} and SURF descriptor: {:.3f}'.format(detector,avg_theta_errors_surf[1]))
        print('Avg. Theta Error for Image 2 with {} and HOG  descriptor: {:.3f}'.format(detector,avg_theta_errors_hog[1]))

        print('Avg. Scale Error for Image 3 with {} and SURF descriptor: {:.3f}'.format(detector,avg_scale_errors_surf[2]))
        print('Avg. Scale Error for Image 3 with {} and HOG  descriptor: {:.3f}'.format(detector,avg_scale_errors_hog[2]))

        print('Avg. Theta Error for Image 3 with {} and SURF descriptor: {:.3f}'.format(detector,avg_theta_errors_surf[2]))
        print('Avg. Theta Error for Image 3 with {} and HOG  descriptor: {:.3f}'.format(detector,avg_theta_errors_hog[2]))
        print()

# author:nannan
# contact: zhaozhaoran@bupt.edu.cn
# datetime:2020/9/13 7:18 下午
# software: PyCharm

import numpy as np
import utils

def my_run():

    img_root = './data/'
    imgs, feats, K = utils.build_img_info(img_root)
    F, pair, match = utils.build_F_pair_match(feats)
    # F, H, pair, match = utils.build_F_H_pair_match(feats)
    # points, edges, tracks, G = utils.extract_points_edges_tracks_G(pair, len(feats))
    img_index1, img_index2 = 0,1
    pts1 = pair[(img_index1, img_index2)]['pts1']
    pts2 = pair[(img_index1, img_index2)]['pts2']
    K1 = K[img_index1]
    K2 = K[img_index2]
    E=utils.get_E_from_F(F,K1,K2)
    # init reconstruction
    R1 = np.eye(3, 3)
    T1 = np.zeros((3, 1))
    R2, T2 = utils.extract_R_T(E, K1, R1, T1, K2, pts1, pts2, 5)
    cloud3d = utils.reconstruct(K1, R1, T1, K2, R2, T2, pts1, pts2)
    utils.save_3d(cloud3d, './my_out/mycloud_3D_euclid.ply')
    # np.save('./cloud3d.npy', cloud3d)
    print(cloud3d.shape)
    #utils.draw_v1(cloud3d)
    utils.show3d( './my_out/mycloud_3D_euclid.ply', 'the result')



if __name__=='__main__':
    my_run()
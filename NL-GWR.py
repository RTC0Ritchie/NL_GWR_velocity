import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile as tif
import scipy.ndimage as ndi
from scipy.interpolate import griddata
import importlib
# 重新加载模块


def transform_A(A_field,rad,centers):
    A_new = None
    for i in range(A_field.shape[0]):
        Ai = []
        for j in range(A_field.shape[1]):
            Ai.append(extract_circles(A_field[i,j],rad,centers, isneedmask=False, isneeddyx=False))
        A_new = np.concatenate((A_new, np.array([Ai])), axis=0) if A_new is not None else np.array([Ai])
    return A_new
def extract_circles(a, r,centers=None, isneedmask=True, isneeddyx=True, order=0):
    """
    a   : 2-D array shape (m, n)
    r   : 半径（像素距离）
    order : 插值方式，0=最近邻（默认，速度最快，无额外计算）
    return : (m*n, k) 的 float64 数组，每行按圆内像素顺序展开
    centers:
        None                -> 全网格采样
        [di, dj]            -> 步长采样 (i, j 方向步长)
        (a,2) array/list    -> 直接给出 (i, j) 坐标
    返回:
        values (N, k)  圆内像素值
        mask   (N, k)  True=被置零
    """
    rad = int(np.ceil(r))
    yy, xx = np.mgrid[-rad:rad+1, -rad:rad+1]
    mask_circle = yy**2 + xx**2 <= r*r
    dyx = np.column_stack((yy[mask_circle].ravel(), xx[mask_circle].ravel()))
    k = dyx.shape[0]

    m, n = a.shape
    if centers is None:                               # 全网格
        ci, cj = np.mgrid[0:m, 0:n]
        centers_ij = np.column_stack((ci.ravel(), cj.ravel()))
    elif np.ndim(centers) == 1 and len(centers) == 2:  # 步长
        di, dj = centers
        ci, cj = np.mgrid[0:m:di, 0:n:dj]
        centers_ij = np.column_stack((ci.ravel(), cj.ravel()))
    else:                                             # 直接给定
        centers = np.asarray(centers, dtype=int)
        if centers.ndim != 2 or centers.shape[1] != 2:
            raise ValueError("centers must be (a,2) array")
        centers_ij = centers

    N = centers_ij.shape[0]

    # 3. 构造所有采样坐标 (2, N*k)
    yx_all = dyx[None, :, :] + centers_ij[:, None, :]    # (N,k,2)
    yx_all = yx_all.reshape(-1, 2).T                     # (2,N*k)

    if isneedmask:
        raw_vals = ndi.map_coordinates(
            a.astype(float, copy=False), yx_all,
            order=order, mode='constant', cval=np.nan)

    padded = np.pad(a.astype(float, copy=False), rad,
                    mode='constant', constant_values=0)
    yx_all_pad = yx_all + rad
    fill_vals = ndi.map_coordinates(
        padded, yx_all_pad, order=order, mode='constant', cval=0.0)

    values = fill_vals.reshape(N, k)
    if isneedmask:
        mask = np.isnan(raw_vals).reshape(N, k) | \
               ((yx_all[0] < 0) | (yx_all[0] >= m) |
                (yx_all[1] < 0) | (yx_all[1] >= n)).reshape(N, k)
        if isneeddyx:
            return values, ~mask,dyx,centers_ij
        else:
            return values, ~mask
    if isneeddyx:
        return values,dyx,centers_ij
    else:
        return values
def gaussion_kernel_basis_1(dyx):
    return np.exp(-(dyx[:,0]**2+dyx[:,1]**2)/2)
def get_kernel_matrix_1d(h,kernel_basis,nn_obs):
    return ((kernel_basis**(1/h**2)/h).reshape(1,-1)).repeat(nn_obs,axis=0)
def gaussion_kernel_basis_2(yx_obs_point):
    ds1 = yx_obs_point[:,0].reshape(-1,1)-yx_obs_point[:,0].reshape(1,-1)
    ds2 = yx_obs_point[:,1].reshape(-1,1)-yx_obs_point[:,1].reshape(1,-1)
    return -(ds1**2+ds2**2)/2
def get_kernel_matrix_2d(b,kernel_basis):
    return np.exp(kernel_basis/b**2)/b

def S_HGWR_nonlinear(y_obs_all,A_all,nanmask,kernel_basis,h,weight_spatial_sigma,nonlinear_attri=None,is_need_ith=True):
    # x_all = []
    # nn_nr = kernel_basis.shape[0]
    # nn_obs = y_obs_all.shape[1]
    # weight_matrix_h = get_kernel_matrix_1d(h,kernel_basis,nn_obs) * nanmask
    # N_obs_all, M_obs_all = [], []
    # for i in range(nn_obs):
    #     N_all, M_all = [], []
    #     for k in range(A_all.shape[0]):
    #         tmpA = A_all[k,:,i,:].T * (nanmask[i,:].reshape(-1,1).repeat(A_all.shape[1],axis=1))
    #         if is_need_ith:
    #             ATW = ((weight_spatial_sigma[k,i,:]*weight_matrix_h[i,:]).reshape(-1,1)*tmpA).T
    #             N_all.append(ATW @ tmpA)
    #             M_all.append(ATW @ y_obs_all[k,i,:,:])
    #         else:
    #             tmpA = np.delete(tmpA,nn_nr//2,axis=0)
    #             tmpweight = np.delete(weight_matrix_h[i,:],nn_nr//2)
    #             tmpweight_sigma = np.delete(weight_spatial_sigma[k,i,:],nn_nr//2)
    #             tmpy = np.delete(y_obs_all[k,i,:,:],nn_nr//2,axis=0)
    #             ATW = ((tmpweight_sigma*tmpweight).reshape(-1,1)*tmpA).T
    #             N_all.append(ATW @ tmpA)
    #             M_all.append(ATW @ tmpy)
    #     N = np.sum(N_all,axis=0)
    #     M = np.sum(M_all,axis=0)
    #     x_pre = np.linalg.inv(N) @ M
    #     x_all.append(x_pre)
    #     N_obs_all.append(N)
    #     M_obs_all.append(M)
    ydim, nn_obs, nn_nr, _ = y_obs_all.shape
    weight_matrix_h = get_kernel_matrix_1d(h,kernel_basis,nn_obs) * nanmask
    A = np.einsum('kaji,ji->kjia', A_all, nanmask)
    # print(A.shape)
    y = y_obs_all

    if not is_need_ith:
        mid = nn_nr // 2
        A = np.concatenate([A[:, :, :mid, :], A[:, :, mid + 1:, :]], axis=2)
        y = np.concatenate([y[:, :, :mid, :], y[:, :, mid + 1:, :]], axis=2)
        weight_matrix_h = np.delete(weight_matrix_h, mid, axis=1)
        weight_spatial_sigma = np.delete(weight_spatial_sigma, mid, axis=2)
        nn_nr -= 1

    w = weight_spatial_sigma * weight_matrix_h[None, :, :]   # (ydim, nn_obs, nn_nr)

    w = w[..., None]  # (ydim, nn_obs, nn_nr, 1)

    N_obs_all = np.einsum('kija,kijb,kij->iab', A, A, w[..., 0])   # (nn_obs, xdim, xdim)

    M_obs_all = np.einsum('kija,kij,kij->ia', A, y[..., 0], w[..., 0])[..., None]  # (nn_obs, xdim, 1)
    # print(M_obs_all.shape)
    x_all = np.linalg.solve(N_obs_all , M_obs_all)  # (nn_obs, xdim, 1)
    if nonlinear_attri is None:
        return np.array(x_all)
    elif nonlinear_attri[0] == 'penalty':
        return penalty_solve(x_all,N_obs_all,M_obs_all,nn_obs,*nonlinear_attri[1:])
    elif nonlinear_attri[0] == 'lagrange':
        return lagrange_solve(x_all,N_obs_all,M_obs_all,nn_obs,*nonlinear_attri[1:])
    else:
        raise NotImplementedError('nonlinear_attri not supported')
def penalty_solve(x_init,N_obs_all,M_obs_all,nn_obs,g_func,dg_func,C,lambda_p,max_iter,tol,mu=0,print_info=False):
    N_obs = np.array(N_obs_all)
    M_obs = np.array(M_obs_all)
    x_init_np = np.array(x_init)
    x_all = [x_init_np]
    iter = 0
    older_dxk_norm = 0
    while True:
        iter += 1
        x_tmp = x_all[-1]
        g = g_func(x_tmp)
        dg = dg_func(x_tmp)
        K = N_obs + C*dg@(dg.transpose((0, 2, 1))) + mu*np.array([np.eye(N_obs.shape[-1])]).repeat(nn_obs,axis=0)
        b = M_obs - N_obs@x_tmp - C*g*dg
        dxk = np.linalg.solve(K,b)
        x_new = x_tmp + lambda_p*dxk
        x_all.append(x_new)
        tmpnorm = np.linalg.norm(dxk)
        if print_info:
            print(iter,tmpnorm)
        if abs(tmpnorm-older_dxk_norm)<tol or iter>max_iter:
            break
        older_dxk_norm = tmpnorm
    return x_all[-1]
def lagrange_solve(x_init,N_obs_all,M_obs_all,nn_obs,g_func,dg_func,hg_func,lambda_p,max_iter,tol):
    N_obs = np.array(N_obs_all)
    M_obs = np.array(M_obs_all)
    x_init_np = np.array(x_init)
    x_all = [x_init_np]
    mu_init = np.zeros((nn_obs,1))
    mu_all = [mu_init]
    iter = 0
    older_dxk_norm = 0
    while True:
        iter += 1
        x_tmp = x_all[-1]
        mu_tmp = mu_all[-1]
        g = g_func(x_tmp)
        dg = dg_func(x_tmp)
        hg = hg_func(x_tmp)

        F1 = N_obs @ x_tmp + mu_tmp.reshape((-1,1,1))*dg - M_obs
        F_comp = np.concatenate((F1,g),axis=1)
        J1 = np.concatenate((N_obs + mu_tmp.reshape((-1,1,1))*hg,dg),axis=2)
        J2 = np.concatenate((dg.transpose((0, 2, 1)),np.zeros((nn_obs,1,1))),axis=2)
        J_comp = np.concatenate((J1,J2),axis=1)
        dxk = -np.linalg.solve(J_comp,F_comp)
        x_new = x_tmp + lambda_p*dxk[:,:-1,:]
        mu_new = mu_tmp + lambda_p*dxk[:,-1,:]

        x_all.append(x_new)
        mu_all.append(mu_new)
        tmpnorm = np.linalg.norm(dxk)
        print(iter,tmpnorm)
        if abs(tmpnorm-older_dxk_norm)<tol or iter>max_iter:
            break
        older_dxk_norm = tmpnorm
    return x_all[-1]

def g_func(x):
    return (x[:,0,0]*x[:,3,0]-x[:,1,0]*x[:,2,0]).reshape((-1,1,1))
def dg_func(x):
    return np.array([[x[:,3,0],-x[:,2,0],-x[:,1,0],x[:,0,0]]]).transpose((2, 1, 0))
def hg_func(x):
    return np.array([[[0,0,0,1],
                     [0,0,-1,0],
                     [0,-1,0,0],
                     [1,0,0,0]]]).repeat(x.shape[0],axis=0)

def CV_h(h,y_obs_all,A_all,nanmask,kernel_basis,weight_data_sigma,nonlinear_attri=None):
    x_pre_without_ith = S_HGWR_nonlinear(y_obs_all,A_all,nanmask,kernel_basis,h,weight_data_sigma,nonlinear_attri,is_need_ith=False)
    A_i = A_all[:,:,:,y_obs_all.shape[2]//2].transpose((2,0,1))
    y_pre_ith_all = A_i @ x_pre_without_ith

    y_true = y_obs_all.squeeze()
    y_pre_ith_all = np.array(y_pre_ith_all).squeeze().T
    CV = np.sum((y_true[:,:,y_true.shape[2]//2]-y_pre_ith_all)**2)
    return CV
def CV_b(b,kernel_basis_obs,z_pre):
    weight_matrix_b = get_kernel_matrix_2d(b,kernel_basis_obs)
    S_b = weight_matrix_b / np.sum(weight_matrix_b,axis=1).reshape(-1,1)
    z_pre = np.array(z_pre)
    Sbz = S_b @ z_pre.T              # (obs, y)
    denom = 1.0 - np.diag(S_b)       # (obs,)
    numer = z_pre.T - Sbz            # (obs, y)
    # print(denom)
    resid = numer / denom[:, None]   # (obs, y)
    CV_b = np.sum(resid**2)
    return CV_b
def newton_para_method(x_init,tol,max_iter,lambda_k,dx_standby,func,attri):
    if len(x_init)<3:
        raise ValueError("x_init must have at least 3 elements")
    x_all = np.array(x_init)
    y_all = np.array([func(x_all[i],*attri) for i in range(x_all.shape[0])])
    iter = 0
    while True:
        iter += 1
        xk2,xk1,xk0 = x_all[-1],x_all[-2],x_all[-3]
        yk2,yk1,yk0 = y_all[-1],y_all[-2],y_all[-3]
        # 计算差商
        f01 = (yk1 - yk0) / (xk1 - xk0)
        f12 = (yk2 - yk1) / (xk2 - xk1)

        # 二阶差商
        a = (f12 - f01) / (xk2 - xk0)          # 近似二阶导数/2
        b = f12 + a * (xk2 - xk1)              # 近似一阶导数

        # 抛物线极小点（导数为零）的修正量
        if a>0:
            dxk = -b / (2 * a)
        else:
            dxk = np.sign(b / (2 * a))*dx_standby/lambda_k
        xk3 = xk2 + lambda_k*dxk
        if xk3 < 0:
            tmp_lambda_k = lambda_k
            while xk3 < 0:
                tmp_lambda_k *= 0.5
                xk3 = xk2 + tmp_lambda_k*dxk
                if tmp_lambda_k<1e-10:
                    raise ValueError("lambda_k is too small")
        x_all = np.append(x_all,xk3)
        if abs(dxk)<tol:
            break
        if iter>=max_iter:
            print("max_iter reached")
            break
        # print(xk3)
        y_all = np.append(y_all,func(xk3,*attri))
    return x_all,y_all

def S_HGWR_KI_CV(y_field_all,rad,centers,A_all,
                 h_init_k0,b_init_k0,
                 h_attri,b_attri,
                 x_attri,
                 sigma_known_k=None,sigma_known_field=None,
                 nonlinear_attri=None,
                 isonly_gwr=False,
                 is_test_h=False,h_range=None,is_test_b=False,b_range=None):
    weight_spatial_field = []
    if sigma_known_k is None:
        weight_spatial_field = np.ones(y_field_all.shape)
    else:
        for k in range(A_all.shape[0]):
            if k in sigma_known_k:
                weight_spatial_field.append(1/sigma_known_field[sigma_known_k.index(k)]**2)
            else:
                weight_spatial_field.append(np.ones(y_field_all[0].shape))
        weight_spatial_field = np.array(weight_spatial_field)

    y_obs_all, weight_spatial_init = [], []
    mask_nan, dyx, yx_obs_point = None, None, None
    for k in range(A_all.shape[0]):
        if k==0:
            y_tmp, mask_nan,dyx,yx_obs_point = extract_circles(y_field_all[k], rad,centers, isneedmask=True, isneeddyx=True)
        else:
            y_tmp, mask_test_tmp = extract_circles(y_field_all[k], rad,centers, isneedmask=True, isneeddyx=False)
            mask_nan = np.logical_and(mask_nan,mask_test_tmp)
        weight_spatial_init.append(extract_circles(weight_spatial_field[k],rad,centers, isneedmask=False, isneeddyx=False))
        y_obs_all.append(y_tmp[:,:,None])
    y_obs_all = np.array(y_obs_all)*(mask_nan[None,:,:,None].repeat(A_all.shape[0],axis=0))
    weight_spatial_init = np.array(weight_spatial_init)*(mask_nan[None,:,:].repeat(A_all.shape[0],axis=0))

    if np.ndim(centers) == 1 and len(centers) == 2:
        stridei,stridej = centers[0],centers[1]
        sd_ni,sd_nj = (y_field_all.shape[1]-1)//stridei+1, (y_field_all.shape[2]-1)//stridej+1
        sd_srcy,sd_srcx = np.arange(y_field_all.shape[1])/float(stridei), np.arange(y_field_all.shape[2])/float(stridej)
        sd_srcY,sd_srcX = np.meshgrid(sd_srcy, sd_srcx, indexing='ij')
    elif np.ndim(centers) == 2:
        sd_srcY,sd_srcX = np.mgrid[0:y_field_all.shape[1],0:y_field_all.shape[2]]
        sd_ni,sd_nj = None,None
    else:
        sd_ni,sd_nj,sd_srcY,sd_srcX = None,None,None,None

    kernel_basis_1 = gaussion_kernel_basis_1(dyx)
    if not isonly_gwr:
        kernel_basis_2 = gaussion_kernel_basis_2(yx_obs_point)
        # print(kernel_basis_2)
        # return None

    h_adaptive = h_attri[0]
    b_adaptive = b_attri[0]
    if h_adaptive:
        dh_init_step,dh_standby,lambda_h,max_iter_h,tol_h = h_attri[1:]
    if b_adaptive:
        db_init_step,db_standby,lambda_b,max_iter_b,tol_b = b_attri[1:]
    max_iter_all, tol_all, test_hb_iter = x_attri

    weight_all = [weight_spatial_init]
    h_all = np.array([h_init_k0])
    b_all = np.array([b_init_k0])
    x_init = S_HGWR_nonlinear(y_obs_all,A_all,mask_nan,kernel_basis_1,h_init_k0,weight_spatial_init,nonlinear_attri,is_need_ith=True)
    # return x_init,None,None,None

    x_all = [x_init]

    iter_all = 0
    A_i = A_all[:,:,:,y_obs_all.shape[2]//2].transpose((2,0,1))

    while True:
        if isonly_gwr:
            break
        iter_all += 1

        h_tmp = h_all[-1]
        b_tmp = b_all[-1]

        # print(A_i.shape,x_all[-1].shape)

        epsi_all = (y_obs_all[:,:,y_obs_all.shape[2]//2,:].transpose((1,0,2)) - A_i @ x_all[-1]).squeeze().T
        z_all = np.array(epsi_all)**2

        kb_matrix = get_kernel_matrix_2d(b_tmp,kernel_basis_2) #(n_obs,n_obs)
        kb_sum = np.sum(kb_matrix,axis=0)[None,:]
        ydim, n_obs = z_all.shape
        if sigma_known_k is None:
            mask_sigma = np.ones((ydim, n_obs), dtype=bool)
        else:
            known = np.asarray(sigma_known_k)
            mask_sigma = np.ones(ydim, dtype=bool)
            mask_sigma[known] = False          # 这些 k 不参与
            mask_sigma = mask_sigma[:, None].repeat(n_obs, axis=1)
        denom = z_all @ kb_matrix.T        # (ydim, n_obs)
        weight_new_single = np.zeros((ydim, n_obs))
        weight_new_single[mask_sigma] = (kb_sum / denom)[mask_sigma]
        weight_new_field,weight_new = [], []
        if centers is None:
            for k in range(A_all.shape[0]):
                if sigma_known_k is not None and k in sigma_known_k:
                    weight_new_field.append(weight_spatial_field[k,:,:])
                else:
                    weight_new_field.append(weight_new_single[k,:].reshape(y_field_all[0,:,:].shape))
        elif np.ndim(centers) == 1 and len(centers) == 2:
            for k in range(A_all.shape[0]):
                if sigma_known_k is not None and k in sigma_known_k:
                    weight_new_field.append(weight_spatial_field[k,:,:])
                else:
                    tmp_weight_new_field = weight_new_single[k,:].reshape(sd_ni,sd_nj)
                    tmp_weight_new_field = ndi.map_coordinates(tmp_weight_new_field, [sd_srcY,sd_srcX], order=1, mode='nearest')
                    weight_new_field.append(tmp_weight_new_field)
        elif np.ndim(centers) == 2:
            for k in range(A_all.shape[0]):
                if sigma_known_k is not None and k in sigma_known_k:
                    weight_new_field.append(weight_spatial_field[k,:,:])
                else:
                    tmp_weight_new_vals = weight_new_single[k,:]
                    out = griddata(centers, tmp_weight_new_vals, (sd_srcY, sd_srcX), method='linear', fill_value=np.nan)
                    out[np.isnan(out)] = griddata(centers, tmp_weight_new_vals, (sd_srcY[np.isnan(out)], sd_srcX[np.isnan(out)]),
                                                  method='nearest')
                    weight_new_field.append(out)
        else:
            raise ValueError("Unsupported input for centers")

        for k in range(A_all.shape[0]):
            if sigma_known_k is not None and k in sigma_known_k:
                weight_new.append(weight_spatial_init[k,:,:])
            else:
                weight_new.append(extract_circles(weight_new_field[k],rad,centers, isneedmask=False, isneeddyx=False))
        weight_tmp = np.array(weight_new)*(mask_nan[None,:,:].repeat(A_all.shape[0],axis=0))
        weight_all.append(weight_tmp)

        if is_test_h:
            print('h')
            # h_range = np.arange(0.1,10,0.1)
            cvh_all = [CV_h(h,y_obs_all,A_all,mask_nan,kernel_basis_1,weight_tmp,nonlinear_attri) for h in h_range]
            plt.figure(figsize=(6,5))
            plt.plot(h_range,cvh_all)
            plt.show()
            break

        if h_adaptive:
            h_init = [h_tmp,h_tmp+dh_init_step,h_tmp-dh_init_step]
            h_newton,_ = newton_para_method(h_init,tol_h,max_iter_h,lambda_h,dh_standby,
                                            CV_h,(y_obs_all,A_all,mask_nan,kernel_basis_1,weight_tmp,nonlinear_attri))
            h_best = h_newton[-1]
            # 防震荡和爆炸
            if h_all.shape[0] > test_hb_iter:
                if h_best > 10*rad:
                    raise ValueError("h_best is too large")
                if np.linalg.norm(np.array([h_all[-2],h_all[-4],h_all[-6]])-h_best)<tol_h:
                    h_best = (h_best+h_all[-1])/2
            h_all = np.append(h_all,h_best)
        else:
            h_best = h_tmp

        x_tmp = S_HGWR_nonlinear(y_obs_all,A_all,mask_nan,kernel_basis_1,h_best,weight_tmp,nonlinear_attri,is_need_ith=True)
        x_all.append(x_tmp)
        # print(np.linalg.norm(x_tmp-x_all[-2]))
        if np.linalg.norm(x_tmp-x_all[-2]) < tol_all:
            break

        if is_test_b:
            print('b')
            # b_range = np.arange(0.1,10,0.1)
            cvb_all = [CV_b(b,kernel_basis_2,z_all) for b in b_range]
            plt.figure(figsize=(6,5))
            plt.plot(b_range,cvb_all)
            plt.show()
            break

        if b_adaptive:
            b_init = [b_tmp,b_tmp+db_init_step,b_tmp-db_init_step]
            b_newton,_ = newton_para_method(b_init,tol_b,max_iter_b,lambda_b,db_standby,
                                            CV_b,(kernel_basis_2,z_all))
            b_best = b_newton[-1]
            if b_all.shape[0] > test_hb_iter:
                if b_best > 10*rad:
                    raise ValueError("b_best is too large")
                if np.linalg.norm(np.array([b_all[-2],b_all[-4],b_all[-6]])-b_best)<tol_b:
                    b_best = b_best + db_init_step*np.sign(b_best-b_all[-1])
            b_all = np.append(b_all,b_best)
        else:
            b_best = b_tmp

        if iter_all >= max_iter_all:
            print("max_iter_all reached")
        print('iter_all:',iter_all)
        print('h:',h_best)
        print('b:',b_best)
        print('resi:',np.linalg.norm(x_tmp-x_all[-2]))
    x_single = x_all[-1]
    weight_single = weight_all[-1]
    x_field = []; weight_field = []
    if centers is None:
        for l in range(x_single.shape[1]):
            x_field.append(x_single[:,l,:].reshape(y_field_all[0].shape))
        for k in range(A_all.shape[0]):
            if sigma_known_k is not None and k in sigma_known_k:
                weight_field.append(weight_spatial_field[k,:,:])
            else:
                weight_field.append(weight_single[k,:,weight_single.shape[2]//2].reshape(y_field_all[0].shape))
    elif np.ndim(centers) == 1 and len(centers) == 2:
        for l in range(x_single.shape[1]):
            tmp_x_field = x_single[:,l,:].reshape(sd_ni,sd_nj)
            tmp_x_field = ndi.map_coordinates(tmp_x_field, [sd_srcY,sd_srcX], order=1, mode='nearest')
            x_field.append(tmp_x_field)
        for k in range(A_all.shape[0]):
            if sigma_known_k is not None and k in sigma_known_k:
                weight_field.append(weight_spatial_field[k,:,:])
            else:
                tmp_weight_field = weight_single[k,:,weight_single.shape[2]//2].reshape(sd_ni,sd_nj)
                tmp_weight_field = ndi.map_coordinates(tmp_weight_field, [sd_srcY,sd_srcX], order=1, mode='nearest')
                weight_field.append(tmp_weight_field)
    elif np.ndim(centers) == 2:
        for l in range(x_single.shape[1]):
            out = griddata(centers, x_single[:,l,0], (sd_srcY, sd_srcX), method='linear', fill_value=np.nan)
            out[np.isnan(out)] = griddata(centers, x_single[:,l,0], (sd_srcY[np.isnan(out)], sd_srcX[np.isnan(out)]),
                                          method='nearest')
            x_field.append(out)
        for k in range(A_all.shape[0]):
            if sigma_known_k is not None and k in sigma_known_k:
                weight_field.append(weight_spatial_field[k,:,:])
            else:
                out = griddata(centers, weight_single[k,:,weight_single.shape[2]//2], (sd_srcY, sd_srcX), method='linear', fill_value=np.nan)
                out[np.isnan(out)] = griddata(centers, weight_single[k,:,weight_single.shape[2]//2], (sd_srcY[np.isnan(out)], sd_srcX[np.isnan(out)]),
                                              method='nearest')
                weight_field.append(out)
    else:
        raise ValueError("Unsupported input for centers")
    return np.array(x_field),np.array(weight_field),h_all[-1],b_all[-1]

if __name__ == '__main__':
    vr_ft = tif.imread(r'preprocessData\ft_vr.tif')
    vaz_ft = tif.imread(r'preprocessData\ft_vaz.tif')
    vr_inf = tif.imread(r'preprocessData\int_vr.tif')
    vx = tif.imread(r'preprocessData\s2_vx.tif')
    vy = tif.imread(r'preprocessData\s2_vy.tif')


    theta1 = tif.imread(r'preprocessData\theta.tif')
    theta1[np.isnan(theta1)] = 0
    # pre.print_v(theta1 * 180 / np.pi, -180,180)
    costheta = np.cos(theta1)
    sintheta = np.sin(theta1)
    onesarray = np.ones(theta1.shape)
    zerosarray = np.zeros(theta1.shape)
    A1 = np.array([[costheta, sintheta, zerosarray, zerosarray]])
    A2 = np.array([[costheta, sintheta, zerosarray, zerosarray]])
    A3 = np.array([[-sintheta, costheta, zerosarray, zerosarray]])
    A4 = np.array([[onesarray, zerosarray, onesarray, zerosarray]])
    A5 = np.array([[zerosarray, onesarray, zerosarray, onesarray]])
    A_double = np.concatenate((A1, A2, A3, A4, A5), axis=0)

    y_all = np.concatenate(([vr_inf], [vr_ft], [vaz_ft], [vx], [vy]), axis=0)
    sigma_vr_inf = np.ones_like(vr_inf) * 0.2
    sigma_vr_ft = np.ones_like(vr_ft) * 5
    sigma_vaz_ft = np.ones_like(vaz_ft) * 2
    sigma_vx = np.ones_like(vx) * 0.5
    sigma_vy = np.ones_like(vy) * 0.5
    sigma_map = np.concatenate(([sigma_vr_inf], [sigma_vr_ft], [sigma_vaz_ft], [sigma_vx], [sigma_vy]), axis=0)

    rad = 5
    stride = [5,5]
    h = 5
    lambda_p = 10000
    A_double_all = np.array(transform_A(A_double,rad,stride))
    x_norm,_,_,_ = S_HGWR_KI_CV(np.array(y_all),rad,stride,A_double_all,
                               h,2,
                               [False,0.1,0.1,0.25,100,1e-4],[False,0.1,0.1,0.25,100,1e-4],
                               [100,1,100],
                               [0,1,2,3,4],sigma_map,
                               ('penalty',g_func,dg_func,lambda_p,0.5,100,1e-3,0,True),
                               True,
                               False,None,False,None)

    tif.imwrite(r'output\vx.tif',x_norm[0])
    tif.imwrite(r'output\vy.tif',x_norm[1])

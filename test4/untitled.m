clear; clc; close all;

M = 4;                  
sigma_theory = 0.45*M;  
fprintf('理论最优 σ = %.2f\n', sigma_theory);

N = 256;
I_chess = checkerboard(8, N/16, N/16) > 0.5;
[x,y] = meshgrid(linspace(0,1,N), linspace(0,1,N));
I_chirp = sin(2*pi*40*(x.^2 + y.^2));

I_chess_direct = I_chess(1:M:end, 1:M:end);
I_chirp_direct = I_chirp(1:M:end, 1:M:end);

sigma = sigma_theory;
h = fspecial('gaussian', round(6*sigma)+1, sigma);
I_chess_filt = imfilter(I_chess, h, 'replicate');
I_chirp_filt = imfilter(I_chirp, h, 'replicate');

I_chess_down = I_chess_filt(1:M:end, 1:M:end);
I_chirp_down = I_chirp_filt(1:M:end, 1:M:end);

figure(1);
subplot(2,3,1); imshow(I_chess,[]); title('棋盘格原图');
subplot(2,3,2); imshow(I_chess_direct,[]); title('直接下采样(混叠)');
subplot(2,3,3); imshow(I_chess_down,[]); title('滤波后下采样');

subplot(2,3,4); imshow(I_chirp,[]); title('Chirp原图');
subplot(2,3,5); imshow(I_chirp_direct,[]); title('直接下采样(混叠)');
subplot(2,3,6); imshow(I_chirp_down,[]); title('滤波后下采样');
sgtitle('3.1 混叠现象与抗混叠滤波对比');

I_camera = im2double(imread('cameraman.tif'));
sigmas = [0.5, 1.0, 1.8, 2.0, 4.0];
figure(2);
for i = 1:length(sigmas)
    s = sigmas(i);
    h = fspecial('gaussian', round(6*s)+1, s);
    If = imfilter(I_camera, h, 'replicate');
    Id = If(1:M:end, 1:M:end);
    subplot(2,3,i); imshow(Id,[]);
    title(sprintf('σ=%.1f',s));
end
sgtitle('4.1 不同σ下采样效果');

[Gx, Gy] = gradient(I_camera);
grad = sqrt(Gx.^2 + Gy.^2);

sigma_min = 0.5;
sigma_max = 2.2;

block = 8;
[h, w] = size(I_camera);
I_adap = zeros(h, w, 'like', I_camera); 

for i = 1:block:h
    for j = 1:block:w
        i2 = min(i+block-1, h);
        j2 = min(j+block-1, w);       
        block_grad = mean(grad(i:i2, j:j2), 'all'); 
        s = sigma_min + (sigma_max - sigma_min) * (block_grad / max(grad(:)));
        hf = fspecial('gaussian', round(6*s) + 1, s);
        I_adap(i:i2, j:j2) = imfilter(I_camera(i:i2, j:j2), hf, 'replicate');
    end
end

sigma_theory = 0.45 * M;
h_global = fspecial('gaussian', round(6*sigma_theory) + 1, sigma_theory);
I_global = imfilter(I_camera, h_global, 'replicate');

I_adap_down = I_adap(1:M:end, 1:M:end);
I_global_down = I_global(1:M:end, 1:M:end);
err = abs(I_global_down - I_adap_down);

figure(3);
subplot(1,3,1); imshow(I_global_down,[]); title('全局固定σ=1.8');
subplot(1,3,2); imshow(I_adap_down,[]); title('自适应σ下采样');
subplot(1,3,3); imshow(err,[]); title('误差图');
colormap jet; colorbar;
sgtitle('5.1 自适应 vs 全局固定滤波');

fprintf('\n===== 思考题答案 =====\n');
fprintf('1. 人脸与背景不应使用相同σ\n');
fprintf('2. 人脸：梯度大、高频多 → σ 偏大(≈%.2f)\n', sigma_theory);
fprintf('3. 背景：平滑、低频多 → σ 偏小(%.1f~%.1f)\n', sigma_min, 1.2);
fprintf('4. 方法：根据局部梯度/方差自适应分配σ\n');
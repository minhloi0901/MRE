def compute_MRE(
    pipeline,
    init_images: torch.Tensor,
    device: torch.device,
    num_masks: int,
    blur_factor: float,
    patch_size=(64, 64),
    seed: int = 0,
):
    N, C, W, H = init_images.size()
    image_size = (init_images.size(2), init_images.size(3))
    rng = torch.Generator(device).manual_seed(seed)

    patch_dims = (
        (image_size[0] + patch_size[0] - 1) // patch_size[0],
        (image_size[1] + patch_size[1] - 1) // patch_size[1],
    )
    ids_per_mask = (patch_dims[0] * patch_dims[1] + num_masks - 1) // num_masks
    s = set()
    masks = [
        [torch.zeros(image_size, dtype=torch.uint8) for _ in range(N)]
        for _ in range(num_masks)
    ]
    for b in range(N):
        ids = torch.randperm(
            patch_dims[0] * patch_dims[1], generator=rng, device=device
        )
    
        for ptr, id in enumerate(ids):
            k = ptr // ids_per_mask
    
            patch_x = id // patch_dims[1]
            patch_y = id % patch_dims[1]
            for i in range(
                patch_x * patch_size[0], (patch_x + 1) * patch_size[0]
            ):
                for j in range(
                    patch_y * patch_size[1], (patch_y + 1) * patch_size[1]
                ):
                    if i < image_size[0] and j < image_size[1]:
                        s.add((k, b))
                        masks[k][b][i, j] = 255
    # alter_mask = torch.zeros((2, *image_size), device=device)
    # for i in range(image_size[0]):
    #     for j in range(image_size[1]):
    #         patch_x = i // patch_size[0]
    #         patch_y = j // patch_size[1]
    #         if (patch_x + patch_y) % 2:
    #             alter_mask[0, i, j] = 1
    #         else:
    #             alter_mask[1, i, j] = 1
    # masks = alter_mask[:, None, :, :].repeat((1, N, 1, 1))

    blurred_masks = [[None for _ in range(N)] for _ in range(num_masks)]
    for k in range(num_masks):
        for b in range(N):
            mask = transforms.ToPILImage()(masks[k][b])
            blurred_masks[k][b] = transforms.ToTensor()(
                pipeline.mask_processor.blur(mask, blur_factor=blur_factor)
            ).to(device)

    images = init_images.clone()
   # save initial images
    for i in range(len(images)):
        pil_image = transforms.ToPILImage()(images[i])
        pil_image.save(f"{i}-init.png")
    for id, mask in enumerate(blurred_masks):
        tmp = pipeline(
            prompt=["" for _ in range(N)],
            image=images,
            mask_image=mask,
            generator=rng,
        ).images
        for i in range(len(tmp)):
            images[i] = transforms.ToTensor()(tmp[i])
        # save current images with blurred mask
        for i in range(len(images)):
            pil_image = transforms.ToPILImage()(images[i])
            pil_image.save(f"{i}-{id}.png")

    return torch.abs(images - init_images)
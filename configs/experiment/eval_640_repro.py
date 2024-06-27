# eval param for different scale

eval_params = dict(
    default = dict(
        img_size=640,
        test_load_size=638,
        letterbox_return_int=True,
        scale_exact=True,
        force_no_pad=True,
        not_infer_on_rect=True,
    ),
    MAFYOLOs = dict(
        img_size=640,
        test_load_size=638,
        letterbox_return_int=True,
        scale_exact=True,
        force_no_pad=True,
        not_infer_on_rect=True,
    ),
    MAFYOLOm = dict(
        img_size=640,
        test_load_size=630,
        letterbox_return_int=True,
        scale_exact=True,
        force_no_pad=True,
        not_infer_on_rect=True,
    ),

)

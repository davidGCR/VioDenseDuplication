def s3d_transform(snippet):
    ''' stack & noralization '''
    # snippet = np.concatenate(snippet, axis=-1)
    # snippet = torch.from_numpy(snippet).permute(2, 0, 1).contiguous().float()
    snippet = snippet.float()
    snippet = snippet.mul_(2.).sub_(255).div(255)

    # out = snippet.view(1,-1,3,snippet.size(1),snippet.size(2)).permute(0,2,1,3,4)
    snippet = snippet.permute(3,0,1,2)

    return snippet
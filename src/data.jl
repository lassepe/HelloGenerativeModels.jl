function table(dataset; static_annotations...)
    [(; x=d[1], y=d[2], static_annotations...) for d in eachcol(dataset)]
end

function decoder_gt(z)
    tanh.(1.5z)
end

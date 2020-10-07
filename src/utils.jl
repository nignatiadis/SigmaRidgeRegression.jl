function _prod_diagonals!(Y, A, B)
    @inbounds for j ∈ 1:size(A, 1)
        Y[j] = 0
        @inbounds for i ∈ 1:size(A, 2)
            Y[j] += A[j, i] * B[i, j]
        end
    end
    Y
end

function random_rotation(p)
    mat = randn(p, p)
    qr(mat).Q
end

struct BlockedSchurComplementSolver{Tf<:AbstractFloat,TA,TB,TC,TS,TSF,TAiu,Tsync,Ttimer}
    A_factorization::TA
    B::TB
    C::TC
    schur_complement::TS
    schur_complement_factorization::TSF
    Ainv_dot_u::TAiu
    synchronize_shared::Tsync
    timer::Ttimer

    function BlockedSchurComplementSolver(
                 dimensions::Vector{<:Dimension}, level::Integer, level_info,
                 schur_complement_buffer_list, second_last_schur_complement_buffer,
                 schur_complement_factorization, use_shared_blocks::Bool,
                 sparse_C_blocks::Bool, shared_comm, synchronize_shared::Fsync,
                 allocate_shared_float::Faf, allocate_shared_int::Fai,
                 block_synchronize_shared::Fbsync, block_allocate_shared_float::Fbaf,
                 block_allocate_shared_int::Fbai, right_multiplication_buffer_storage,
                 C_dense_buffer_storage,
                 check_lu::Bool) where {Fsync,Faf,Fai,Fbsync,Fbaf,Fbai}

        if shared_comm == MPI.COMM_NULL
            # This process should do no work
            null_A_factorization = MPIStaticCondensationNull{Float64}()
            null_B = nothing
            null_C = nothing
            null_schur_complement = nothing
            null_schur_complement_factorization = MPIStaticCondensationNull{Float64}()
            null_Ainv_dot_u = nothing
            null_timer = nothing
            return new{Float64,typeof(null_A_factorization),typeof(null_B),typeof(null_C),typeof(null_schur_complement),typeof(null_schur_complement_factorization),typeof(null_Ainv_dot_u),Fsync,typeof(null_timer)}(
                   null_A_factorization, null_B, null_C, null_schur_complement,
                   null_schur_complement_factorization, null_Ainv_dot_u,
                   synchronize_shared, null_timer)
        end

        if isa(schur_complement_factorization, MPIStaticCondensationNull)
            timer = nothing
        else
            timer = schur_complement_factorization.timer
        end

        shared_comm_rank = MPI.Comm_rank(shared_comm)
        shared_comm_size = MPI.Comm_size(shared_comm)
        block_comm = level_info.block_comm
        if block_comm == MPI.COMM_NULL
            block_comm_rank = 0
            block_comm_size = 1
        else
            block_comm_rank = MPI.Comm_rank(block_comm)
            block_comm_size = MPI.Comm_size(block_comm)
        end

        n_hypercube_positions = 2^sum(level_info.nblock .> 1)

@sc_timeit timer "level $level schur_complement_buffer" begin
        if level ≤ length(schur_complement_buffer_list)
            this_sc_buffer = schur_complement_buffer_list[level]
        else
            this_sc_buffer = second_last_schur_complement_buffer
        end
        schur_complement = BlockS(this_sc_buffer, level_info.local_bottom_vector_indices,
                                  shared_comm, allocate_shared_float)
end

        if level == 1 || !sparse_C_blocks
            matrix_template = nothing
        else
            matrix_template = schur_complement_buffer_list[level-1]
        end

        data_type = eltype(schur_complement.matrix)
        nbottom = length(level_info.local_bottom_vector_indices)

        if use_shared_blocks
            if level_info.block_comm == MPI.COMM_NULL
                A_factorization = MPIStaticCondensationNull{data_type}()
                B = nothing
                C = nothing
            else
@sc_timeit timer "level $level get_block_diagonal_solver" begin
                A_factorization = get_block_diagonal_solver(level_info, data_type,
                                                            level==1, use_shared_blocks,
                                                            timer, check_lu,
                                                            block_allocate_shared_float,
                                                            block_allocate_shared_int,
                                                            block_synchronize_shared)
end
@sc_timeit timer "level $level Ainv_dot_B_buffer" begin
                B = BlockAinvDotBShared{data_type}(level_info.local_top_vector_a_block_indices[1],
                                                   level_info.a_block_off_diagonal_indices[1],
                                                   level_info.a_block_off_diagonal_bottom_vector_indices[1],
                                                   block_comm_rank, block_comm_size,
                                                   block_allocate_shared_float,
                                                   block_synchronize_shared)
end
@sc_timeit timer "level $level C setup" begin
                C_vector_intermediate_buffer =
                    allocate_shared_float(n_hypercube_positions, nbottom)
                if shared_comm_rank == 0
                    C_vector_intermediate_buffer .= 0.0
                end
                C_vector_points_per_proc = (nbottom + shared_comm_size - 1) ÷ shared_comm_size
                C_vector_range = shared_comm_rank*C_vector_points_per_proc+1:min((shared_comm_rank+1)*C_vector_points_per_proc,nbottom)
                C_block_row_inds_full = level_info.a_block_off_diagonal_indices[1]

               C_nrow = length(level_info.a_block_off_diagonal_indices[1])
               C_rows_per_proc = (C_nrow + block_comm_size - 1) ÷ block_comm_size
               if isempty(level_info.local_top_vector_a_block_indices[1])
                   # There are no entries in the block handled by this process, so
                   # to avoid accessing zero-length vectors, set the row range to
                   # be empty also.
                   C_partial_row_range = 1:0
               else
                   C_partial_row_range = block_comm_rank*C_rows_per_proc+1:min((block_comm_rank+1)*C_rows_per_proc,C_nrow)
               end

                block_hypercube_position =
                    get_hypercube_position(level_info.iblock_list[:,1], level_info.nblock)

end
@sc_timeit timer "level $level C creation" begin
#println("level=$level")
#println("level_info.a_block_off_diagonal_indices[1]=$(level_info.a_block_off_diagonal_indices[1])")
#println("C_partial_row_range=$C_partial_row_range")
#println("C_vector_intermediate_buffer=$C_vector_intermediate_buffer")
#println("C_vector_range=$C_vector_range")
#println("C_vector_init_range=$C_vector_init_range")
#println("C_matrix_init_range=$C_matrix_init_range")
#println("level_info.subgroup_i=", level_info.subgroup_i)
                C = BlockCShared{data_type}(level_info.a_block_off_diagonal_indices[1],
                                            level_info.a_block_off_diagonal_bottom_vector_indices[1],
                                            C_partial_row_range,
                                            level_info.local_top_vector_a_block_indices[1],
                                            level_info.local_top_vector_indices,
                                            level_info.local_bottom_vector_indices,
                                            matrix_template, block_hypercube_position,
                                            n_hypercube_positions,
                                            right_multiplication_buffer_storage,
                                            C_dense_buffer_storage,
                                            C_vector_intermediate_buffer, C_vector_range,
                                            level_info.subgroup_i,
                                            block_allocate_shared_float,
                                            block_synchronize_shared, block_comm_rank,
                                            block_comm_size, synchronize_shared)
end
            end
        else
@sc_timeit timer "level $level get_block_diagonal_solver" begin
            A_factorization = get_block_diagonal_solver(level_info, data_type, level==1,
                                                        false, timer, check_lu)
end
@sc_timeit timer "level $level Ainv_dot_B_buffer" begin
            B = BlockAinvDotBSerial{data_type}(level_info.local_top_vector_a_block_indices,
                                               level_info.a_block_off_diagonal_indices,
                                               level_info.a_block_off_diagonal_bottom_vector_indices)
end
@sc_timeit timer "level $level C setup" begin
            C_vector_intermediate_buffer =
                allocate_shared_float(n_hypercube_positions, nbottom)
            if shared_comm_rank == 0
                C_vector_intermediate_buffer .= 0.0
            end
            C_vector_points_per_proc = (nbottom + shared_comm_size - 1) ÷ shared_comm_size
            C_vector_range = shared_comm_rank*C_vector_points_per_proc+1:min((shared_comm_rank+1)*C_vector_points_per_proc,nbottom)

            block_hypercube_positions =
                [get_hypercube_position(iblock, level_info.nblock)
                 for (iblock, bi) ∈ zip(eachcol(level_info.iblock_list), level_info.local_top_vector_a_block_indices)
                 if !isempty(bi)]

end
@sc_timeit timer "level $level C creation" begin
            C = BlockCSerial{data_type}(level_info.a_block_off_diagonal_indices,
                                        level_info.a_block_off_diagonal_bottom_vector_indices,
                                        level_info.local_top_vector_a_block_indices,
                                        level_info.local_top_vector_indices,
                                        level_info.local_bottom_vector_indices,
                                        matrix_template, block_hypercube_positions,
                                        n_hypercube_positions,
                                        right_multiplication_buffer_storage,
                                        C_dense_buffer_storage,
                                        C_vector_intermediate_buffer, C_vector_range,
                                        block_synchronize_shared, synchronize_shared)
end
        end

        if use_shared_blocks
            # Only one block per process.
            Ainv_dot_u = block_allocate_shared_float(length(level_info.local_top_vector_a_block_indices[1]))
        else
            Ainv_dot_u = [block_allocate_shared_float(length(bi))
                          for bi ∈ level_info.local_top_vector_a_block_indices if !isempty(bi)]
        end

        return new{data_type,typeof(A_factorization),typeof(B),typeof(C),typeof(schur_complement),typeof(schur_complement_factorization),typeof(Ainv_dot_u),Fsync,typeof(timer)}(
               A_factorization, B, C, schur_complement, schur_complement_factorization,
               Ainv_dot_u, synchronize_shared, timer)
    end
end

function lu!(sc::BlockedSchurComplementSolver, full_A)
    A_factorization = sc.A_factorization
    B = sc.B
    C = sc.C
    schur_complement = sc.schur_complement
    schur_complement_factorization = sc.schur_complement_factorization
    synchronize_shared = sc.synchronize_shared
    timer = sc.timer

    @sc_timeit timer "lu! BlockedSchurComplementSolver" begin
        @sc_timeit timer "lu(A)" begin
            lu!(A_factorization, full_A)
        end
        @sc_timeit timer "Ainv_dot_B" begin
            synchronize_shared()

@sc_timeit timer "copy_B_submatrix!" begin
            copy_B_submatrix!(B, full_A)
end

@sc_timeit timer "ldiv_block_Bmatrix!" begin
            ldiv_block_Bmatrix!(A_factorization, B)
end
        end
        @sc_timeit timer "C" begin
            copy_C_submatrix!(C, full_A)
        end
        @sc_timeit timer "schur_complement" begin
@sc_timeit timer "sync 1" begin
            synchronize_shared()
end

@sc_timeit timer "C.(A^-1.B)" begin
            mul_C_Ainv_dot_B!(schur_complement, C, B)
end
@sc_timeit timer "sync 2" begin
            synchronize_shared()
end
@sc_timeit timer "add D to schur_complement" begin
            add_D_to_schur_complement!(schur_complement, full_A)
end
@sc_timeit timer "sync 3" begin
            synchronize_shared()
end

@sc_timeit timer "parallel LU schur_complement" begin
            lu!(schur_complement_factorization, schur_complement.matrix)
end
        end
    end

    return nothing
end

function ldiv!(X::AbstractVector, y::AbstractVector, sc::BlockedSchurComplementSolver,
               U::AbstractVector, v::AbstractVector)
    @inbounds begin
        schur_complement_factorization = sc.schur_complement_factorization
        synchronize_shared = sc.synchronize_shared
        timer = sc.timer

        @sc_timeit timer "ldiv!" begin
            A_factorization = sc.A_factorization
            B = sc.B
            C = sc.C
            Ainv_dot_u = sc.Ainv_dot_u
            bottom_sub_range = C.vector_range

            @sc_timeit timer "Ainv.u" begin
                ldiv!(Ainv_dot_u, A_factorization, U)
                synchronize_shared()
            end

            @sc_timeit timer "v-C.Ainv.u" begin
                mul_C_dot_Ainv_dot_u!(y, C, Ainv_dot_u)

                for i ∈ bottom_sub_range
                    y[i] += v[i]
                end
                synchronize_shared()
            end

            @sc_timeit timer "Sinv.(v-C.Ainv.u)" begin
                ldiv!(schur_complement_factorization, y)
                synchronize_shared()
                # MPIStaticCondensation takes care of copying the entries from `y` back
                # into `X`.
            end

            @sc_timeit timer "Ainv.u-Ainv.B.y" begin
                Ainv_dot_u_minus_Ainv_dot_B_dot_y!(X, Ainv_dot_u, B, y)
            end
        end

        return nothing
    end
end

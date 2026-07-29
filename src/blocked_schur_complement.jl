function extract_block_field_from_Tuple(t::Tuple, fieldname::Symbol)
    # `t` is a Tuple of structs, length nt. One of the fields of the struct, `fieldname`,
    # is a Vector, length nv. We want to return a Vector of nv Tuples of nt values from
    # `fieldname`.
    nt = length(t)
    if nt < 1
        error("Tuple `t=$t` has no entries.")
    end
    first_field = getfield(t[1], fieldname)
    nv = length(first_field)
    fieldtype = eltype(first_field)

    return NTuple{nt,fieldtype}[Tuple(getfield(t[it], fieldname)[iv]
                                      for it ∈ 1:nt) for iv ∈ 1:nv]
end

struct BlockedSchurComplementSolver{Tf<:AbstractFloat,TA,TB,TC,TS,TSF,TAiu,Tsync,Ttimer}
    A_factorization::TA
    B::TB
    C::TC
    schur_complement::TS
    schur_complement_solver::TSF
    Ainv_dot_u::TAiu
    synchronize_shared::Tsync
    timer::Ttimer

    function BlockedSchurComplementSolver(
                 dimensions::Vector{<:Dimension}, level::Integer, level_info,
                 schur_complement_buffer_list, second_last_schur_complement_buffer,
                 schur_complement_solver, use_shared_blocks::Bool,
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
            null_schur_complement_solver = MPIStaticCondensationNull{Float64}()
            null_Ainv_dot_u = nothing
            null_timer = nothing
            return new{Float64,typeof(null_A_factorization),typeof(null_B),typeof(null_C),typeof(null_schur_complement),typeof(null_schur_complement_solver),typeof(null_Ainv_dot_u),Fsync,typeof(null_timer)}(
                   null_A_factorization, null_B, null_C, null_schur_complement,
                   null_schur_complement_solver, null_Ainv_dot_u,
                   synchronize_shared, null_timer)
        end

        if isa(schur_complement_solver, MPIStaticCondensationNull)
            timer = nothing
        else
            timer = schur_complement_solver.timer
        end

        shared_comm_rank = MPI.Comm_rank(shared_comm)
        shared_comm_size = MPI.Comm_size(shared_comm)
        block_comm = level_info[1].block_comm
        if block_comm == MPI.COMM_NULL
            block_comm_rank = 0
            block_comm_size = 1
        else
            block_comm_rank = MPI.Comm_rank(block_comm)
            block_comm_size = MPI.Comm_size(block_comm)
        end

        n_hypercube_positions = 2^sum(level_info[1].nblock .> 1)

        if level ≤ length(schur_complement_buffer_list)
            this_sc_buffer = schur_complement_buffer_list[level]
            schur_complement = BlockS(this_sc_buffer,
                                      Tuple(li.local_bottom_vector_indices for li ∈ level_info),
                                      shared_comm, allocate_shared_float)
            data_type = eltype(schur_complement.matrix[1][1])
        else
            schur_complement = BlockDenseS(second_last_schur_complement_buffer,
                                           Tuple(li.local_bottom_vector_indices for li ∈ level_info),
                                           shared_comm, allocate_shared_float)
            data_type = eltype(schur_complement.matrix)
        end

        if level == 1 || !sparse_C_blocks
            matrix_template = nothing
        else
            matrix_template = schur_complement_buffer_list[level-1]
        end

        nbottom = sum(length(li.local_bottom_vector_indices) for li ∈ level_info)

        if use_shared_blocks
            if block_comm == MPI.COMM_NULL
                A_factorization = MPIStaticCondensationNull{data_type}()
                B = nothing
                C = nothing
            else
                A_factorization = get_block_diagonal_solver(level_info, data_type,
                                                            use_shared_blocks, timer,
                                                            check_lu,
                                                            block_allocate_shared_float,
                                                            block_allocate_shared_int,
                                                            block_synchronize_shared)
                B = BlockAinvDotBShared{data_type}(
                        Tuple(li.local_top_vector_a_block_indices[1] for li ∈ level_info),
                        Tuple(li.local_top_vector_a_block_offset_indices[1] for li ∈ level_info),
                        Tuple(li.a_block_off_diagonal_indices[1] for li ∈ level_info),
                        Tuple(li.a_block_off_diagonal_bottom_vector_indices[1] for li ∈ level_info),
                        block_comm_rank, block_comm_size, block_allocate_shared_float,
                        block_synchronize_shared)
                C_vector_intermediate_buffer =
                    allocate_shared_float(n_hypercube_positions, nbottom)
                if shared_comm_rank == 0
                    C_vector_intermediate_buffer .= 0.0
                end
                C_vector_points_per_proc = (nbottom + shared_comm_size - 1) ÷ shared_comm_size
                C_vector_range = shared_comm_rank*C_vector_points_per_proc+1:min((shared_comm_rank+1)*C_vector_points_per_proc,nbottom)

                block_hypercube_position =
                    get_hypercube_position(level_info[1].iblock_list[:,1], level_info[1].nblock)

                C = BlockCShared{data_type}(Tuple(li.a_block_off_diagonal_indices[1] for li ∈ level_info),
                                            Tuple(li.a_block_off_diagonal_bottom_vector_indices[1] for li ∈ level_info),
                                            Tuple(li.a_block_off_diagonal_bottom_vector_offset_indices[1] for li ∈ level_info),
                                            Tuple(li.local_top_vector_a_block_indices[1] for li ∈ level_info),
                                            matrix_template, block_hypercube_position,
                                            n_hypercube_positions,
                                            right_multiplication_buffer_storage,
                                            C_dense_buffer_storage,
                                            C_vector_intermediate_buffer, C_vector_range,
                                            level_info[1].subgroup_i,
                                            block_allocate_shared_float,
                                            block_synchronize_shared, block_comm_rank,
                                            block_comm_size, synchronize_shared)
            end
        else
            A_factorization = get_block_diagonal_solver(level_info, data_type, false,
                                                        timer, check_lu)
            B = BlockAinvDotBSerial{data_type}(
                    extract_block_field_from_Tuple(level_info, :local_top_vector_a_block_indices),
                    extract_block_field_from_Tuple(level_info, :local_top_vector_a_block_offset_indices),
                    extract_block_field_from_Tuple(level_info, :a_block_off_diagonal_indices),
                    extract_block_field_from_Tuple(level_info, :a_block_off_diagonal_bottom_vector_indices))
            C_vector_intermediate_buffer =
                allocate_shared_float(n_hypercube_positions, nbottom)
            if shared_comm_rank == 0
                C_vector_intermediate_buffer .= 0.0
            end
            C_vector_points_per_proc = (nbottom + shared_comm_size - 1) ÷ shared_comm_size
            C_vector_range = shared_comm_rank*C_vector_points_per_proc+1:min((shared_comm_rank+1)*C_vector_points_per_proc,nbottom)

            block_hypercube_positions =
                [get_hypercube_position(iblock, level_info[1].nblock)
                 for (iblock, bi) ∈ zip(eachcol(level_info[1].iblock_list), level_info[1].local_top_vector_a_block_indices)
                 if !isempty(bi)]

            C = BlockCSerial{data_type}(
                    extract_block_field_from_Tuple(level_info, :a_block_off_diagonal_indices),
                    extract_block_field_from_Tuple(level_info, :a_block_off_diagonal_bottom_vector_indices),
                    extract_block_field_from_Tuple(level_info, :a_block_off_diagonal_bottom_vector_offset_indices),
                    extract_block_field_from_Tuple(level_info, :local_top_vector_a_block_indices),
                    matrix_template, block_hypercube_positions, n_hypercube_positions,
                    right_multiplication_buffer_storage, C_dense_buffer_storage,
                    C_vector_intermediate_buffer, C_vector_range,
                    block_synchronize_shared, synchronize_shared)
        end

        if use_shared_blocks
            # Only one block per process.
            Ainv_dot_u = block_allocate_shared_float(sum(length(li.local_top_vector_a_block_indices[1])
                                                         for li ∈ level_info))
        else
            unfiltered_nblock = length(level_info[1].local_top_vector_a_block_indices)
            Nvar = length(level_info)
            top_block_sizes = [sum(length(level_info[ili].local_top_vector_a_block_indices[ib]) for ili ∈ 1:Nvar)
                               for ib ∈ 1:unfiltered_nblock]
            Ainv_dot_u = [block_allocate_shared_float(nb)
                          for nb ∈ top_block_sizes if nb > 0]
        end

        return new{data_type,typeof(A_factorization),typeof(B),typeof(C),typeof(schur_complement),typeof(schur_complement_solver),typeof(Ainv_dot_u),Fsync,typeof(timer)}(
               A_factorization, B, C, schur_complement, schur_complement_solver,
               Ainv_dot_u, synchronize_shared, timer)
    end
end

function lu!(sc::BlockedSchurComplementSolver, full_A)
    A_factorization = sc.A_factorization
    B = sc.B
    C = sc.C
    schur_complement = sc.schur_complement
    schur_complement_solver = sc.schur_complement_solver
    synchronize_shared = sc.synchronize_shared
    timer = sc.timer

    @sc_timeit timer "lu! BlockedSchurComplementSolver" begin
        @sc_timeit timer "lu(A)" begin
            lu!(A_factorization, full_A)
        end
        @sc_timeit timer "Ainv_dot_B" begin
            synchronize_shared()

            copy_B_submatrix!(B, full_A)

            ldiv_block_Bmatrix!(A_factorization, B)
        end
        @sc_timeit timer "C" begin
            copy_C_submatrix!(C, full_A)
        end
        @sc_timeit timer "schur_complement" begin
            synchronize_shared()

            mul_C_Ainv_dot_B!(schur_complement, C, B)
            synchronize_shared()
            add_D_to_schur_complement!(schur_complement, full_A)
            synchronize_shared()

            lu!(schur_complement_solver, schur_complement.matrix)
        end
    end

    return nothing
end

function ldiv!(X::AbstractVector, y::AbstractVector, sc::BlockedSchurComplementSolver,
               U::AbstractVector, v::AbstractVector)
    @inbounds begin
        schur_complement_solver = sc.schur_complement_solver
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
                ldiv!(schur_complement_solver, y)
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

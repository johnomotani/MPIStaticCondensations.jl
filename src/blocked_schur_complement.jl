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
                 sparse_schur_complement_buffer_list, dense_schur_complement_buffer_list,
                 schur_complement_solver, use_shared_blocks::Bool, sparse_C_blocks::Bool,
                 shared_comm, synchronize_shared::Fsync, allocate_shared_float::Faf,
                 block_synchronize_shared::Fbsync, block_allocate_shared_float::Fbaf,
                 block_allocate_shared_int::Fbai, right_multiplication_buffer_storage,
                 C_dense_buffer_storage, check_lu::Bool,
                 data_type::Type) where {Fsync,Faf,Fbsync,Fbaf,Fbai}

        if shared_comm == MPI.COMM_NULL
            # This process should do no work
            null_A_factorization = MPIStaticCondensationNull{Float64}()
            null_B = nothing
            null_C = nothing
            null_schur_complement = nothing
            null_schur_complement_solver = MPIStaticCondensationNull{Float64}()
            null_Ainv_dot_u = nothing
            null_timer = nothing
            return new{data_type,typeof(null_A_factorization),typeof(null_B),typeof(null_C),typeof(null_schur_complement),typeof(null_schur_complement_solver),typeof(null_Ainv_dot_u),Fsync,typeof(null_timer)}(
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
        block_shared_comm = level_info[1].block_shared_comm
        if block_shared_comm == MPI.COMM_NULL
            block_shared_comm_rank = 0
            block_shared_comm_size = 1
        else
            block_shared_comm_rank = MPI.Comm_rank(block_shared_comm)
            block_shared_comm_size = MPI.Comm_size(block_shared_comm)
        end

        if level ≤ length(sparse_schur_complement_buffer_list)
            this_sc_buffer = sparse_schur_complement_buffer_list[level]
            schur_complement = BlockS(this_sc_buffer,
                                      Tuple(li.local_bottom_vector_indices for li ∈ level_info),
                                      shared_comm, allocate_shared_float)
        else
            dense_level_count = level - length(sparse_schur_complement_buffer_list)
            schur_complement = BlockDenseS(dense_schur_complement_buffer_list[dense_level_count],
                                           Tuple(li.local_bottom_vector_indices for li ∈ level_info),
                                           shared_comm, allocate_shared_float)
        end

        if level == 1 || !sparse_C_blocks
            matrix_template = nothing
        else
            matrix_template = sparse_schur_complement_buffer_list[level-1]
        end

        nbottom = sum(length(li.local_bottom_vector_indices) for li ∈ level_info)

        if use_shared_blocks
            if block_shared_comm == MPI.COMM_NULL
                A_factorization = MPIStaticCondensationNull{data_type}()
                B = nothing
                C = NullBlockCShared(level_info[1].n_subgroups,
                                     sum(length(li.local_bottom_vector_indices) for li ∈ level_info),
                                     shared_comm, shared_comm_rank, shared_comm_size,
                                     synchronize_shared, allocate_shared_float)
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
                        Tuple(li.a_block_off_diagonal_bottom_vector_offset_indices[1] for li ∈ level_info),
                        block_shared_comm_rank, block_shared_comm_size, block_allocate_shared_float,
                        block_synchronize_shared)
                C = BlockCShared{data_type}(Tuple(li.a_block_off_diagonal_indices[1] for li ∈ level_info),
                                            Tuple(li.a_block_off_diagonal_bottom_vector_indices[1] for li ∈ level_info),
                                            Tuple(li.a_block_off_diagonal_bottom_vector_offset_indices[1] for li ∈ level_info),
                                            Tuple(li.local_top_vector_a_block_indices[1] for li ∈ level_info),
                                            sum(length(li.local_bottom_vector_indices) for li ∈ level_info),
                                            matrix_template,
                                            right_multiplication_buffer_storage,
                                            C_dense_buffer_storage,
                                            level_info[1].subgroup_i,
                                            level_info[1].n_subgroups,
                                            block_allocate_shared_float,
                                            isa(schur_complement, BlockDenseS),
                                            block_shared_comm_rank, block_shared_comm_size,
                                            shared_comm, shared_comm_rank,
                                            shared_comm_size, synchronize_shared,
                                            allocate_shared_float)
            end
        else
            A_factorization = get_block_diagonal_solver(level_info, data_type, false,
                                                        timer, check_lu)
            B = BlockAinvDotBSerial{data_type}(
                    extract_block_field_from_Tuple(level_info, :local_top_vector_a_block_indices),
                    extract_block_field_from_Tuple(level_info, :local_top_vector_a_block_offset_indices),
                    extract_block_field_from_Tuple(level_info, :a_block_off_diagonal_indices),
                    extract_block_field_from_Tuple(level_info, :a_block_off_diagonal_bottom_vector_offset_indices))
            C = BlockCSerial{data_type}(
                    extract_block_field_from_Tuple(level_info, :a_block_off_diagonal_indices),
                    extract_block_field_from_Tuple(level_info, :a_block_off_diagonal_bottom_vector_indices),
                    extract_block_field_from_Tuple(level_info, :a_block_off_diagonal_bottom_vector_offset_indices),
                    extract_block_field_from_Tuple(level_info, :local_top_vector_a_block_indices),
                    sum(length(li.local_bottom_vector_indices) for li ∈ level_info),
                    matrix_template, right_multiplication_buffer_storage,
                    C_dense_buffer_storage, shared_comm, shared_comm_rank,
                    shared_comm_size, synchronize_shared, allocate_shared_float)
        end

        if use_shared_blocks
            # Only one block per process.
            if length(level_info[1].local_top_vector_a_block_indices) == 0
                Ainv_dot_u = zeros(data_type, 0)
            else
                Ainv_dot_u = block_allocate_shared_float(sum(length(li.local_top_vector_a_block_indices[1])
                                                             for li ∈ level_info))
            end
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

            add_D_to_schur_complement!(schur_complement, full_A)
            synchronize_shared()
            mul_C_Ainv_dot_B!(schur_complement, C, B)

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
            bottom_sub_range = C.this_proc_bottom_vector_entries

            @sc_timeit timer "Ainv.u" begin
                ldiv!(Ainv_dot_u, A_factorization, U)
                synchronize_shared()
            end

            @sc_timeit timer "v-C.Ainv.u" begin
                for i ∈ bottom_sub_range
                    y[i] = v[i]
                end

                mul_C_dot_Ainv_dot_u!(y, C, Ainv_dot_u)
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

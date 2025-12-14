library ieee;
use ieee.std_logic_1164.all;

entity tb_reg is
end entity;

architecture sim of tb_reg is

    constant N : integer := 8;

    signal clk   : std_logic := '0';
    signal rst_a : std_logic := '0';
    signal rst_s : std_logic := '0';
    signal d     : std_logic_vector(N-1 downto 0) := (others => '0');
    signal q     : std_logic_vector(N-1 downto 0);

begin

    --  generic register
    UUT : entity work.main_register
        generic map(
            N => N
        )
        port map(
            clk   => clk,
            rst_a => rst_a,
            rst_s => rst_s,
            d     => d,
            q     => q
        );

    -- clock generation
    clk <= not clk after 10 ns;

 
    process
    begin
        -- ASYNC RESET
        rst_a <= '1';
        wait for 30 ns;
        rst_a <= '0';

        -- LOAD FIRST VALUE AA 
        d <= x"AA";
        wait for 60 ns;

        -- SYNC RESET
        rst_s <= '1';
        wait for 20 ns;
        rst_s <= '0';

        -- LOAD SECOND VALUE CC
        d <= x"CC";
        wait for 60 ns;

        wait;
    end process;

end architecture;
